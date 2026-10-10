#!/bin/bash
# AutoMorph 全流程一键脚本（CPU 友好 + 断点续跑）
# 用法: bash run_automorph_full.sh

set -e

export AUTOMORPH_DATA=/root/autodl-tmp/AI4C/AutoMorph_results
export PYTHONPATH=.:$PYTHONPATH
cd /root/autodl-tmp/AI4C/AutoMorph-main

mkdir -p $AUTOMORPH_DATA/images $AUTOMORPH_DATA/Results

# 确保软链存在
if [ ! -L $AUTOMORPH_DATA/images/eyepacs ]; then
  ln -sf /root/autodl-tmp/AI4C/AutoMorph-main/input/eyepacs $AUTOMORPH_DATA/images/eyepacs
fi
[ -f $AUTOMORPH_DATA/resolution_information.csv ] || cp resolution_information.csv $AUTOMORPH_DATA/
[ -f $AUTOMORPH_DATA/name_mapping.csv ] || cp name_mapping.csv $AUTOMORPH_DATA/

echo "==================================="
echo " AutoMorph 流水线开始 $(date)"
echo " AUTOMORPH_DATA=$AUTOMORPH_DATA"
echo "==================================="

# ===== M0 预处理 =====
if [ ! -d $AUTOMORPH_DATA/Results/M0 ] || [ -z "$(ls -A $AUTOMORPH_DATA/Results/M0/images 2>/dev/null)" ]; then
  echo "[M0] 预处理开始 $(date)"
  cd M0_Preprocess
  python EyeQ_process_main.py 2>&1 | tee $AUTOMORPH_DATA/Results/M0.log
  cd ..
else
  echo "[M0] 已有结果，跳过"
fi

# ===== M1 质量评估 =====
if [ ! -f $AUTOMORPH_DATA/Results/M1/Retinal_quality.csv ]; then
  echo "[M1] 质量评估开始 $(date)"
  cd M1_Retinal_Image_quality_EyePACS
  CUDA_VISIBLE_DEVICES="" sh test_outside.sh 2>&1 | tee $AUTOMORPH_DATA/Results/M1.log
  python merge_quality_assessment.py 2>&1 | tee -a $AUTOMORPH_DATA/Results/M1.log
  cd ..
else
  echo "[M1] 已有结果，跳过"
fi

# ===== M2 分割 =====
if [ ! -f $AUTOMORPH_DATA/Results/M2/disc_cup.json ]; then
  echo "[M2] 血管分割开始 $(date)"
  cd M2_Vessel_seg
  CUDA_VISIBLE_DEVICES="" sh test_outside.sh 2>&1 | tee $AUTOMORPH_DATA/Results/M2_vessel.log
  cd ..

  echo "[M2] 动静脉分割开始 $(date)"
  cd M2_Artery_vein
  CUDA_VISIBLE_DEVICES="" sh test_outside.sh 2>&1 | tee $AUTOMORPH_DATA/Results/M2_av.log
  cd ..

  echo "[M2] 视盘视杯分割开始 $(date)"
  cd M2_lwnet_disc_cup
  CUDA_VISIBLE_DEVICES="" sh test_outside.sh 2>&1 | tee $AUTOMORPH_DATA/Results/M2_disc.log
  cd ..
else
  echo "[M2] 已有结果，跳过"
fi

# ===== M3 特征提取 =====
echo "[M3] 特征提取开始 $(date)"
cd M3_feature_zone/retipy/
python create_datasets_disc_centred_B.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_discB.log
python create_datasets_disc_centred_C.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_discC.log
python create_datasets_macular_centred_B.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_macB.log
python create_datasets_macular_centred_C.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_macC.log
cd ../..

cd M3_feature_whole_pic/retipy/
python create_datasets_macular_centred.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_whole_mac.log
python create_datasets_disc_centred.py 2>&1 | tee $AUTOMORPH_DATA/Results/M3_whole_disc.log
cd ../..

# ===== 合并 =====
python csv_merge.py 2>&1 | tee $AUTOMORPH_DATA/Results/csv_merge.log

echo "==================================="
echo " 完成 $(date)"
echo " 结果: $AUTOMORPH_DATA/Results/M3/"
echo "==================================="
