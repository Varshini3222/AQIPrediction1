# Run_ML_models_FIXED.py
import subprocess
import sys
import os
import time

print("🚀 RUNNING ALL ML HYBRID MODELS...")
print("=" * 80)

# List of all hybrid models with .py extension
hybrid_scripts = [
    'hybrid_xgb_rf.py', 'hybrid_xgb_gb.py', 'hybrid_xgb_dt.py', 'hybrid_xgb_knn.py',
    'hybrid_rf_gb.py', 'hybrid_rf_dt.py', 'hybrid_rf_knn.py', 'hybrid_gb_dt.py',
    'hybrid_gb_knn.py', 'hybrid_dt_knn.py', 'hybrid_xgb_rf_gb.py', 'hybrid_xgb_rf_dt.py',
    'hybrid_xgb_rf_knn.py', 'hybrid_xgb_gb_dt.py', 'hybrid_xgb_gb_knn.py', 'hybrid_xgb_dt_knn.py',
    'hybrid_rf_gb_dt.py', 'hybrid_rf_gb_knn.py', 'hybrid_rf_dt_knn.py', 'hybrid_gb_dt_knn.py',
    'hybrid_xgb_rf_gb_dt.py', 'hybrid_xgb_rf_gb_knn.py', 'hybrid_rf_gb_dt_knn.py',
    'hybrid_gb_dt_knn_xgb.py', 'hybrid_xgb_rf_dt_knn.py', 'hybrid_xgb_rf_gb_dt_knn.py'
]

# Alternative naming if you used different names
alternative_names = [
    'XGB_RF.py', 'XGB_GB.py', 'XGB_DT.py', 'XGB_KNN.py',
    'RF_GB.py', 'RF_DT.py', 'RF_KNN.py', 'GB_DT.py',
    'GB_KNN.py', 'DT_KNN.py', 'XGB_RF_GB.py', 'XGB_RF_DT.py',
    'XGB_RF_KNN.py', 'XGB_GB_DT.py', 'XGB_GB_KNN.py', 'XGB_DT_KNN.py',
    'RF_GB_DT.py', 'RF_GB_KNN.py', 'RF_DT_KNN.py', 'GB_DT_KNN.py',
    'XGB_RF_GB_DT.py', 'XGB_RF_GB_KNN.py', 'RF_GB_DT_KNN.py',
    'GB_DT_KNN_XGB.py', 'XGB_RF_DT_KNN.py', 'XGB_RF_GB_DT_KNN.py'
]

successful_runs = 0
failed_runs = 0

for i, script in enumerate(alternative_names, 1):
    print(f"\n{'=' * 70}")
    print(f"[{i}/26] EXECUTING: {script}")
    print(f"{'=' * 70}")

    # Check if file exists
    if not os.path.exists(script):
        print(f"❌ FILE NOT FOUND: {script}")
        failed_runs += 1
        continue

    try:
        # Run with timeout and capture output
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # Print ALL output
        if result.stdout:
            print("📊 MODEL OUTPUT:")
            print(result.stdout)

        if result.stderr:
            print("⚠️  WARNINGS/ERRORS:")
            print(result.stderr)

        print(f"⏱️  Execution time: {execution_time:.2f} seconds")

        if result.returncode == 0:
            print(f"✅ SUCCESS: {script} completed")
            successful_runs += 1
        else:
            print(f"❌ FAILED: {script} returned code {result.returncode}")
            failed_runs += 1

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {script} took too long (>5 minutes)")
        failed_runs += 1
    except Exception as e:
        print(f"💥 ERROR: {script} - {str(e)}")
        failed_runs += 1

print("\n" + "=" * 80)
print("📈 FINAL SUMMARY")
print("=" * 80)
print(f"✅ Successful: {successful_runs}/26")
print(f"❌ Failed: {failed_runs}/26")
print(f"📊 Success Rate: {(successful_runs / 26) * 100:.1f}%")

if successful_runs > 0:
    print("\n🎉 Hybrid models completed! Check the generated:")
    print("   - Confusion matrix images (.png files)")
    print("   - Saved model files (.joblib files)")
    print("   - Performance metrics in output above")