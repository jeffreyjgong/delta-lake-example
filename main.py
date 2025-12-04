import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import SparkSession
from delta import *

# --- Configuration ---
import pathlib
base_dir = pathlib.Path(__file__).parent.resolve()
table_path = str(base_dir / "tmp" / "delta_identity_demo")
warehouse_dir = str(base_dir / "tmp" / "warehouse")

# Clean up previous runs
if os.path.exists(table_path):
    shutil.rmtree(table_path)
if os.path.exists(warehouse_dir):
    shutil.rmtree(warehouse_dir)

print("=" * 70)
print("TEST: Concurrent writes to Delta table with IDENTITY columns")
print("=" * 70)
print("\nExpected result: CONCURRENT WRITES SHOULD FAIL")
print("Identity columns disable concurrent transactions in Delta Lake.\n")

# --- Initialize Spark ---
builder = SparkSession.builder \
    .appName("DeltaConcurrentIdentityTest") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.warehouse.dir", warehouse_dir)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

try:
    # --- Create table with identity column ---
    print("Creating Delta table with IDENTITY column...")
    from delta.tables import DeltaTable, IdentityGenerator
    from pyspark.sql.types import LongType, StringType
    
    DeltaTable.create(spark) \
        .addColumn("id", LongType(), nullable=False, 
                   generatedAlwaysAs=IdentityGenerator(start=1, step=1)) \
        .addColumn("writer", StringType()) \
        .addColumn("data", StringType()) \
        .location(table_path) \
        .execute()
    
    print("Table created.\n")

    # --- Attempt concurrent writes ---
    NUM_WRITERS = 3
    
    def write_data(writer_id):
        """Each writer tries to insert one row."""
        df = spark.createDataFrame([
            (f"writer_{writer_id}", f"data_from_{writer_id}")
        ], ["writer", "data"])
        df.write.format("delta").mode("append").save(table_path)
        return f"Writer {writer_id} succeeded"

    print(f"Launching {NUM_WRITERS} concurrent writers...")
    print("-" * 70)
    
    successes = 0
    failures = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WRITERS) as executor:
        futures = {executor.submit(write_data, i): i for i in range(NUM_WRITERS)}
        
        for future in as_completed(futures):
            writer_id = futures[future]
            try:
                result = future.result()
                print(f"  ✓ {result}")
                successes += 1
            except Exception as e:
                # Extract just the error code from the full exception
                error_msg = str(e).split('\n')[0][:80]
                print(f"  ✗ Writer {writer_id} FAILED: {error_msg}")
                failures += 1

    # --- Summary ---
    print("-" * 70)
    print(f"\nResults:")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")
    
    # Show what actually got written
    print("\nData in table:")
    df = spark.read.format("delta").load(table_path)
    df.show()
    
    print(f"Total rows: {df.count()}")
    
    if failures > 0:
        print("\n" + "=" * 70)
        print("CONFIRMED: Concurrent writes to identity column tables fail!")
        print("This is expected behavior - identity columns require exclusive writes.")
        print("=" * 70)

finally:
    spark.stop()
    print("\nSpark session stopped.")
