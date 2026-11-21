
Download list

This page tells you where to find all the necessary files for the hackathon.
Riverops wheel

You can find the riverops v0.2.0 wheel at: riverops-0.2.0-py3-none-any.whl
Handgesture recognition dataset

You will find the handgesture recognition dataset at: durance-ai-handgesture-recognition.zip.
Yocto Linux image

In case you destroyed your Linux distribution, you might need to flash it again.

You can find the linux image here: yocto-hackathon-img.gz.

Typical usage below. CAUTION: sudo dd can wipe one of your computer's partitions, make sure to locate the SD card device!

# Find your SD card (e.g. /dev/sda)
lsblk
YOUR_SD_CARD=...

# Decompress and flash
gzip -dc yocto-hackaton.img.gz | sudo dd of="$YOUR_SD_CARD" bs=4M status=progress conv=fsync

A bunch of other links

A bunch of links in no particular order:

    arm_sdk_installer.sh
    go2-kria-generated.zip
    ros-docker-original.zip
    x86_64_sdk_installer.sh

