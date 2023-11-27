#!/usr/bin/env bash
function gdrive_download () { # credit to https://github.com/ethanjperez/convince
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

mkdir -p Models/real-fixed-cam Models/real-hand-held
gdrive_download 1yiNsSkPYoBZ55fSQ1iwb1io9QL_PcR2i Models/real-fixed-cam/netG_epoch_12.pth
gdrive_download 13HckO9fPAKYocdB_CAC5n8uyM3xQ2MpG Models/real-hand-held/netG_epoch_12.pth