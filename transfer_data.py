import os
import argparse
import subprocess

def main(args):
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    
    # Download data file from GCS
    if args.gcs_bucket:
        print(f"Downloading data from gs://{args.gcs_bucket}/{args.data_file}")
        subprocess.run([
            "gsutil", "cp", 
            f"gs://{args.gcs_bucket}/{args.data_file}", 
            f"data/{args.data_file}"
        ])
        
        # Download images
        print(f"Downloading images from gs://{args.gcs_bucket}/images/")
        subprocess.run([
            "gsutil", "cp", "-r",
            f"gs://{args.gcs_bucket}/images/*", 
            "data/images/"
        ])
    else:
        print("No GCS bucket specified. Please upload data manually.")
        print("You can use:")
        print("  gsutil cp your-local-file.csv gs://your-bucket/")
        print("  gsutil cp -r your-local-images/ gs://your-bucket/images/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer data from GCS to VM")
    parser.add_argument("--gcs_bucket", type=str, help="GCS bucket name")
    parser.add_argument("--data_file", type=str, default="antm2c_10m_part0", help="Data file name")
    args = parser.parse_args()
    
    main(args)