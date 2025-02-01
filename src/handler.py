'''
Handler for the generation of a fine tuned lora model.
'''

import os
import shutil
import subprocess

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA

def handler(job):
    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']

    model_url = job_input['model_url']
    model_basename = os.path.basename(model_url)

    VOLUME_DIR = "/runpod-volume"

    # Check if model exists in volume directory
    volume_model_path = os.path.join(VOLUME_DIR, model_basename)
    if os.path.exists(volume_model_path):
        print(f"Model found in volume, using cached version: {volume_model_path}")
        downloaded_model = {'file_path': volume_model_path}
    else:
        # Download the model file
        print(f"Downloading model from {model_url}")
        downloaded_model = rp_download.file(job_input['model_url'])

        # Make sure we check if the volume directory exists, in that case just use the download file path
        if os.path.exists(VOLUME_DIR):
            print(f"Moving model to volume for caching: {volume_model_path}")
            shutil.copy(downloaded_model['file_path'], volume_model_path)
            original_file_path = downloaded_model['file_path']

            # Update the file path to the volume directory
            downloaded_model['file_path'] = volume_model_path

            # Delete old file
            os.remove(original_file_path)

    # Download the zip file
    print(f"Downloading zip file from {job_input['zip_url']}")
    downloaded_input = rp_download.file(job_input['zip_url'])

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(
            f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}"
    os.makedirs(flat_directory, exist_ok=True)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

    out_id = job_input['out_id'] or job['id']
    subprocess.run(f"""accelerate launch --num_cpu_threads_per_process=1 sdxl_train_network.py \
                         --enable_bucket \
                         --pretrained_model_name_or_path={downloaded_model['file_path']} \
                         --train_data_dir="./training/img" \
                         --resolution=1024,1024 \
                         --network_alpha=1 \
                         --text_encoder_lr=5e-05 \
                         --no_half_vae \
                         --mixed_precision='fp16' \
                         --save_precision='fp16' \
                         --full_fp16 \
                         --gradient_checkpointing \
                         --unet_lr={job_input['unet_lr']} \
                         --network_dim={job_input['network_dim']} \
                         --lr_scheduler={job_input['lr_scheduler']} \
                         --learning_rate={job_input['learning_rate']} \
                         --lr_scheduler_num_cycles={job_input['lr_scheduler_num_cycles']} \
                         --lr_warmup_steps={job_input['lr_warmup_steps']} \
                         --train_batch_size={job_input['train_batch_size']} \
                         --max_train_steps={job_input['max_train_steps']} \
                         --output_dir="./training/model" \
                         --output_name={out_id} \
                         --max_data_loader_n_workers={job_input['max_data_loader_num_workers']} \
                         --save_model_as=safetensors \
                         --network_module=networks.lora \
                         --optimizer_type {job_input['optimizer_type']} \
                         --cache_latents --bucket_reso_steps=64 --bucket_no_upscale""", shell=True, check=True)

    job_s3_config = job.get('s3Config')

    uploaded_lora_url = upload_file_to_bucket(
        file_name=f"{out_id}.safetensors",
        file_location=f"./training/model/{out_id}.safetensors",
        bucket_creds=job_s3_config,
        bucket_name=None if job_s3_config is None else job_s3_config['bucketName'],
    )

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})
