#!/usr/bin/env python3
# medium_customer_360.py
# Purpose: Build a customer-360 daily snapshot with features: RFM metrics, category affinities, CLV proxies.
# Uses: adaptive execution, broadcast hints, skew handling, DPP, and a Java UDF (from JAR).
# How to run:
# spark-submit --packages io.delta:delta-core_2.12:2.4.0 \
#   --jars /opt/jars/acme-udfs-1.0.jar \
#   medium_customer_360.py \
#   --env prod --run_date 2026-03-01 --country IN --full_refresh false --dpp true --skew true

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType, DoubleType

def parse_args():
    p = argparse.ArgumentParser(description="Customer 360 Medium Complexity")
    p.add_argument("--env", required=True, choices=["dev","test","prod"])
    p.add_argument("--run_date", required=True)
    p.add_argument("--country", default="IN")
    p.add_argument("--full_refresh", default="false", choices=["true","false"])
    p.add_argument("--dpp", default="true", choices=["true","false"], help="Enable dynamic partition pruning")
    p.add_argument("--skew", default="true", choices=["true","false"], help="Enable AQE skew join handling")
    p.add_argument("--sample_ratio", type=float, default=0.0)
    p.add_argument("--output_table", default="curated.customer_360_daily")
    return p.parse_args()

def build_spark(app_name: str, args) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")  # we will use explicit hints
        .config("spark.sql.shuffle.partitions", "600" if args.env == "prod" else "80")
        .config("spark.sql.optimizer.dynamicPartitionPruning", args.dpp)
        .config("spark.sql.adaptive.skewJoin.enabled", args.skew)
        .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
        .getOrCreate()
    )

    # Register Java UDFs from custom JAR
    spark.udf.registerJavaFunction("mask_email", "com.acme.udf.MaskingUDF.maskEmail", StringType())
    spark.udf.registerJavaFunction("normalize_text", "com.acme.udf.TextUDF.normalize", StringType())

    return spark

def main():
    args = parse_args()
    spark = build_spark("medium_customer_360", args)

    # Base tables
    customers  = spark.table("db.dim_customers").where(F.col("country") == args.country)
    stores     = spark.table("db.dim_stores").where(F.col("country") == args.country)
    products   = spark.table("db.dim_products")
    categories = spark.table("db.dim_categories")
    orders     = spark.table("db.fact_orders").where(F.to_date("order_date") <= F.lit(args.run_date))
    items      = spark.table("db.fact_order_items")
    payments   = spark.table("db.fact_payments")

    if args.sample_ratio and args.sample_ratio > 0:
        customers = customers.sample(args.sample_ratio)
        orders    = orders.sample(args.sample_ratio)
        items     = items.sample(args.sample_ratio)
        payments  = payments.sample(args.sample_ratio)

    # Normalize email and names (using Java UDF + builtin)
    customers_norm = (
        customers
        .withColumn("email_norm", F.expr("mask_email(email)"))
        .withColumn("name_norm", F.expr("normalize_text(customer_name)"))
        .select(
            "customer_id","name_norm","email_norm","city","state","country","signup_date","phone"
        )
    )

    # Enrich items with categories
    prod_enriched = (
        products.alias("p")
        .join(F.broadcast(categories.alias("c")), F.col("p.category_id")==F.col("c.category_id"), "left")
        .select("p.product_id","p.product_name","p.brand","p.category_id","c.category_name")
    )

    # Orders with store info (explicit broadcast for small dims)
    orders_enriched = (
        orders.alias("o")
        .join(F.broadcast(stores.alias("s")), F.col("o.store_id")==F.col("s.store_id"), "left")
        .select(
            "o.order_id","o.customer_id","o.order_date","o.store_id",
            "s.store_city","s.store_name"
        )
    )

    # Line-level amounts & join to product category
    line = (
        items.alias("i")
        .join(orders_enriched.alias("o"), F.col("i.order_id")==F.col("o.order_id"), "inner")
        .join(prod_enriched.alias("p"), F.col("i.product_id")==F.col("p.product_id"), "left")
        .withColumn("extended_price", F.col("i.qty") * F.col("i.unit_price"))
        .select(
            "o.customer_id", "o.order_id", "o.order_date", "o.store_id", "o.store_city",
            "p.product_id", "p.category_id", "p.category_name",
            "i.qty","i.unit_price","extended_price"
        )
    )

    # Payments aggregated at order level
    pay = (
        payments.groupBy("order_id")
        .agg(
            F.sum("paid_amount").alias("paid_amount"),
            F.max("payment_status").alias("payment_status"),
            F.max("payment_method").alias("payment_method")
        )
    )

    # Assemble order-level view
    orders_full = (
        line.groupBy("order_id","customer_id","order_date","store_id","store_city")
        .agg(
            F.sum("extended_price").alias("order_amount"),
            F.countDistinct("product_id").alias("unique_products"),
            F.max("category_name").alias("any_category")  # just an example
        )
        .join(pay, "order_id", "left")
        .withColumn("is_paid", (F.col("paid_amount") >= F.col("order_amount")).cast("boolean"))
    )

    # RFM (Recency, Frequency, Monetary) features
    w_cust_date_desc = Window.partitionBy("customer_id").orderBy(F.col("order_date").desc())
    rfm = (
        orders_full
        .withColumn("rownum", F.row_number().over(w_cust_date_desc))
        .withColumn("recency_days", F.datediff(F.lit(args.run_date), F.max("order_date").over(Window.partitionBy("customer_id"))))
        .groupBy("customer_id")
        .agg(
            F.count("*").alias("frequency_orders"),
            F.sum("order_amount").alias("monetary_total"),
            F.max("recency_days").alias("recency_days")
        )
    )

    # Category affinity (top category by spend)
    cat_affinity = (
        line.groupBy("customer_id","category_name")
        .agg(F.sum("extended_price").alias("category_spend"))
    )
    w_aff = Window.partitionBy("customer_id").orderBy(F.col("category_spend").desc(), F.col("category_name").asc())
    cat_affinity = (
        cat_affinity
        .withColumn("rk", F.row_number().over(w_aff))
        .where(F.col("rk")==1)
        .select("customer_id", F.col("category_name").alias("top_category"), "category_spend")
    )

    # Payment behavior
    pay_behavior = (
        orders_full.groupBy("customer_id")
        .agg(
            F.avg(F.col("paid_amount") - F.col("order_amount")).alias("avg_overpay"),
            F.sum(F.when(F.col("is_paid")==False, 1).otherwise(0)).alias("unpaid_orders"),
            F.count("*").alias("orders_total")
        )
        .withColumn("unpaid_rate", F.col("unpaid_orders")/F.col("orders_total"))
    )

    # Join all features to customers
    features = (
        customers_norm.alias("c")
        .join(rfm.alias("r"), F.col("c.customer_id")==F.col("r.customer_id"), "left")
        .join(cat_affinity.alias("a"), F.col("c.customer_id")==F.col("a.customer_id"), "left")
        .join(pay_behavior.alias("pb"), F.col("c.customer_id")==F.col("pb.customer_id"), "left")
        .select(
            F.col("c.customer_id"),
            "name_norm","email_norm","city","state","country","signup_date",
            "frequency_orders","monetary_total","recency_days",
            "top_category","category_spend",
            "avg_overpay","unpaid_orders","orders_total","unpaid_rate"
        )
        .fillna({"frequency_orders":0,"monetary_total":0.0,"recency_days":9999,
                 "top_category":"UNKNOWN","category_spend":0.0,"avg_overpay":0.0,
                 "unpaid_orders":0,"orders_total":0,"unpaid_rate":0.0})
    )

    # CLV proxy
    features = features.withColumn(
        "clv_proxy",
        (F.col("monetary_total") * 0.2) + (F.col("frequency_orders") * 10) - (F.col("unpaid_rate") * 50)
    )

    # SCD2-ish daily snapshot (simple approach: overwrite for the day)
    snapshot = (
        features
        .withColumn("run_date", F.lit(args.run_date))
        .withColumn("env", F.lit(args.env))
    )

    # QA metrics
    total_customers = customers_norm.count()
    total_features  = snapshot.count()
    coverage = 0.0 if total_customers == 0 else (total_features/total_customers)
    print(f"[METRIC] customers={total_customers}, feature_rows={total_features}, coverage={coverage:.2f}")

    # Write to a managed table
    # You can switch to Delta if desired (ensure --packages delta is passed)
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {args.output_table} (
            customer_id string,
            name_norm string,
            email_norm string,
            city string,
            state string,
            country string,
            signup_date date,
            frequency_orders bigint,
            monetary_total double,
            recency_days int,
            top_category string,
            category_spend double,
            avg_overpay double,
            unpaid_orders bigint,
            orders_total bigint,
            unpaid_rate double,
            clv_proxy double,
            run_date date,
            env string
        )
        USING PARQUET
        PARTITIONED BY (run_date, env)
    """)

    (
        snapshot
        .repartition(1)
        .write
        .mode("overwrite")
        .insertInto(args.output_table)
    )

    # Data quality checks
    null_emails = snapshot.filter(F.col("email_norm").isNull()).count()
    if null_emails > 0:
        print(f"[WARN] Found {null_emails} rows with null masked emails")

    high_risk_unpaid = snapshot.filter(F.col("unpaid_rate") > 0.5).count()
    print(f"[INFO] High-risk customers (unpaid_rate>0.5): {high_risk_unpaid}")

    spark.stop()

if __name__ == "__main__":
    main()
