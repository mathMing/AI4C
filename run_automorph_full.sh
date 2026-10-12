#!/bin/bash
# AutoMorph 全流程一键脚本
set -e
export AUTOMORPH_DATA=/root/autodl-tmp/AI4C/AutoMorph_results
export PYTHONPATH=.:$PYTHONPATH
cd /root/autodl-tmp/AI4C/AutoMorph-main

mkdir -p $AUTOMORPH_DATA/images $AUTOMORPH_DATA/Results
if [ ! -L $AUTOMORPH_DATA/images/eyepacs ]; then
  ln -sf /root/autodl-tmp/AI4C/AutoMorph-main/input/eyepacs $AUTOMORPH_DATA/images/eyepacs
fi
[ -f $AUTOMORPH_DATA/resolution_information.csv ] || cp resolution_information.csv $AUTOMORPH_DATA/
[ -f $AUTOMORPH_DATA/name_mapping.csv ] || cp name_mapping.csv $AUTOMORPH_DATA/

echo "===== 开始 $(date) ====="

# M0
if [ ! -d $AUTOMORPH_DATA/Results/M0/images ] || [ -z "$(ls -A $AUTOMORPH_DATA/Results/M0/images 2>/dev/null)" ]; then
  echo "[M0] $(date)"; cd M0_Preprocess; python EyeQ_process_main.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M0.log; cd ..
fi

# M1
if [ ! -f $AUTOMORPH_DATA/Results/M1/Retinal_quality.csv ]; then
  echo "[M1] $(date)"; cd M1_Retinal_Image_quality_EyePACS; CUDA_VISIBLE_DEVICES="" bash test_outside.sh 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M1.log; python merge_quality_assessment.py 2>&1 | tee -a /root/autodl-tmp/AI4C/AutoMorph_results/M1.log; cd ..
fi

# M2
if [ ! -f $AUTOMORPH_DATA/Results/M2/disc_cup.json ]; then
  echo "[M2 vessel] $(date)"; cd M2_Vessel_seg; CUDA_VISIBLE_DEVICES="" bash test_outside.sh 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M2_vessel.log; cd ..
  echo "[M2 av] $(date)"; cd M2_Artery_vein; CUDA_VISIBLE_DEVICES="" bash test_outside.sh 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M2_av.log; cd ..
  echo "[M2 disc] $(date)"; cd M2_lwnet_disc_cup; CUDA_VISIBLE_DEVICES="" bash test_outside.sh 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M2_disc.log; cd ..
fi

# M3
echo "[M3] $(date)"; cd M3_feature_zone/retipy/; python create_datasets_disc_centred_B.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_discB.log; python create_datasets_disc_centred_C.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_discC.log; python create_datasets_macular_centred_B.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_macB.log; python create_datasets_macular_centred_C.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_macC.log; cd ../..
cd M3_feature_whole_pic/retipy/; python create_datasets_macular_centred.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_whole_mac.log; python create_datasets_disc_centred.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/M3_whole_disc.log; cd ../..

python csv_merge.py 2>&1 | tee /root/autodl-tmp/AI4C/AutoMorph_results/csv_merge.log
echo "===== 完成 $(date) ====="
