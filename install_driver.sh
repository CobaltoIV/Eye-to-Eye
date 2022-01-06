sudo rm /etc/X11/xorg.conf
sudo apt install nvidia-prime
sudo prime-select nvidia
sudo apt install nvidia-driver-460
sudo modprobe nvidia
sudo nvidia-smi -pm 1
