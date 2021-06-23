#!/bin/bash

# Here are the links you requested. Please note the below urls expire in 90 days; if additional time is needed or you would like to download the files again, you will need to submit a new request on https://fastmri.med.nyu.edu:

# Knee MRI:
# knee_singlecoil_train (~88 GB)
# knee_singlecoil_val (~19 GB)
# knee_singlecoil_test (~7 GB)
# knee_singlecoil_challenge (~1.5 GB)
# knee_multicoil_train (~931 GB)
# knee_multicoil_val (~192 GB)
# knee_multicoil_test (~109 GB)
# knee_multicoil_challenge (~16.2 GB)
# knee_DICOMs_batch1 (~134 GB)
# knee_DICOMs_batch2 (~30 GB)

# Brain MRI:
# brain_multicoil_challenge (~36.5 GB)
# brain_multicoil_challenge_transfer (~19.1 GB)
# brain_multicoil_train (~1228.8 GB)
# brain_multicoil_val (~350.9 GB)
# brain_multicoil_test (~34.2 GB)
# brain_fastMRI_DICOM (~39.6 GB)
# SHA256 Hash (0.5 KB)

# To download Knee MRI files, we recommend using curl with recovery mode turned on:

outdir='/home/zhihua/data/public_datasets/fastMRI/'
logdir='/home/zhihua/temp'

echo "Bash version ${BASH_VERSION}..."
echo "Downloading fastMRI data"
echo "---------------------------------"

curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=KxT2958EksRrMASpCwMlsQJB3a0%3D&Expires=1625047397" --output ${outdir}knee_singlecoil_train.tar.gz > ${logdir}1.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=lgE22XIaF0Y3Dj%2BQWuoYNP53T84%3D&Expires=1625047397" --output  ${outdir}knee_singlecoil_val.tar.gz > ${logdir}2.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_test_v2.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WMsVbz240wnP3TSSIOIU5C3EgRI%3D&Expires=1625047397" --output  ${outdir}knee_singlecoil_test_v2.tar.gz > ${logdir}3.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_singlecoil_challenge.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=yNqmP1sMBPWnD%2FLwnMyQgLqeYKs%3D&Expires=1625047397" --output  ${outdir}knee_singlecoil_challenge.tar.gz > ${logdir}4.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=9w94T7pBiSPSIxGAa1P0sYlqI1k%3D&Expires=1625047397" --output  ${outdir}multicoil_train.tar.gz > ${logdir}5.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/multicoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=DMOHwUIytEURupmNSJ%2FroA%2BPM24%3D&Expires=1625047397" --output  ${outdir}multicoil_val.tar.gz > ${logdir}6.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_multicoil_test_v2.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qsxFBSZbM%2BjWjrPFbL9RNe0J51Y%3D&Expires=1625047397" --output  ${outdir}knee_multicoil_test_v2.tar.gz > ${logdir}8.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_multicoil_challenge.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ZtwdMwmP23NHJs6zEStSTOufK2w%3D&Expires=1625047397" --output  ${outdir}knee_multicoil_challenge.tar.gz > ${logdir}9.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_mri_dicom_batch1.tar?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=W2%2BK6m%2Bq%2F3XxjNc3vPa2PbJZVrQ%3D&Expires=1625047397" --output  ${outdir}knee_mri_dicom_batch1.tar > ${logdir}10.log 2>&1 &
curl -C - "https://fastmri-dataset.s3.amazonaws.com/knee_mri_dicom_batch2.tar?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=4YasTbwX9EQPUMVy3yO1Z3QAtuc%3D&Expires=1625047397" --output  ${outdir}knee_mri_dicom_batch2.tar > ${logdir}11.log 2>&1 &

# echo "---------------------------------"

# To download Brain MRI files, we recommend using curl with recovery mode turned on:

# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_challenge.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=YCAUM12hPqXgL0AR6N8G03PrL7E%3D&Expires=1625047397" --output ${outdir}brain_multicoil_challenge.tar.gz > ${logdir}12.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_challenge_transfer.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=9Qi%2B8pd5XORV17td7cymoPUgP%2FM%3D&Expires=1625047397" --output ${outdir}brain_multicoil_challenge_transfer.tar.gz > ${logdir}13.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_train.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=dG%2BUG0II5%2Faj%2FA%2B6ZeJL3K5v6ds%3D&Expires=1625047397" --output ${outdir}brain_multicoil_train.tar.gz > ${logdir}14.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_val.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=IkZNxgtfUSPx521zX5jZBmMyd0M%3D&Expires=1625047397" --output ${outdir}brain_multicoil_val.tar.gz > ${logdir}15.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_multicoil_test.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Ckv41jvhgjh3MRznzsm2dn7WdFs%3D&Expires=1625047397" --output ${outdir}brain_multicoil_test.tar.gz > ${logdir}16.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/brain_fastMRI_DICOM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=KdQIOFiXg8khqYQMAYRw3ONVJfQ%3D&Expires=1625047397" --output ${outdir}brain_fastMRI_DICOM.tar.gz > ${logdir}17.log 2>&1 &
# curl -C - "https://fastmri-dataset.s3.amazonaws.com/SHA256?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=xtIiFLur0dDA8fAPW0fKkhYMOS0%3D&Expires=1625047397" --output ${outdir}SHA256 > ${outdir}18.log 2>&1 &

# For your reference, you can go to https://fastmri.med.nyu.edu for information on how to cite us as well as a copy of the data use agreement.
