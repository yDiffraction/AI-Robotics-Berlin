
A tiny bit of theory...

Overall, this is how you proceed to develop, build and execute ROS2 software :

    (laptop): Modify ROS 2 C++ code on laptop
    (laptop): Build ROS2 packages from inside Docker container (it has all dependencies, but you can skip Docker if you want to install your own dependencies. In that case, check Containerfile for what your need.)
    (laptop): Deploy to Kria through SSH
    (kria): Start ROS 2 application on Kria
    (robot): Application communicates with Go2 via Ethernet cable connected to Robot
    (robot): Go2's internal controller executes commands

If you have read this carefully you might have noticed that we need 2 Ethernet cables. Kria KV260 only has 1 Ethernet port. For development without Robot this is enough, but you need 2 ports when investigating while Robot is online. Normally we provided an Ethernet-to-USB adapter for this.

Theoretically, you don't even need 2 Ethernet cables. 1 Ethernet cable is required to connect to Unitree GO2 robot, but you can access console via UART. When you need to copy files manually, you can do this by removing SD card and inserting it into your PC and mounting the SD card. This process is annoying, so making 2nd Ethernet cable work is preferable for you.

NOTE : We provided a Docker container with necessary dependencies. Containerfile includes the dependencies included in Docker container image. Feel free to add more. Remember to rebuild image afterwards. If you don't want to use Docker, feel free to install all dependencies on your laptop.

IMPORTANT : Docker installation is tricky. Use the preferred official method : https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

FYI, my docker version is 29.7.2, build a7dcaa6
Physical setup
Getting familiar with Kria KV260

    Side 1
        find SD card slot
        find a slot for ribbon camera cable (right next to SD card slot)
        find UART port (on the same side as the previous 2 interfaces)
    Side 2
        find RESET button for rebooting Kria
    Side 3 opposite 1
        Power cable slot
        Screen interfaces (if you ever need them)
        4 USB ports
        Ethernet port

Branching everything

    Insert SD card
    Attach camera
    (Optionally) connect UART port to your laptop (connect using something like "sudo minicom -D /dev/ttyUSB1 -b 115200")
    (Optionally) connect screen via HDMI or whatever
    (During demo) connect Ethernet-USB adapter that will be used to connect to robot
    (Optionally) connect Ethernet port to your laptop
    Connect power cable

If you are on UART, you will immediately see logs of bootloader and then kernel.

If you are on Ethernet, you need to wait till it boots until connecting.

Use whichever of these 2 commands to connect :

# Either via ssh
ssh root@durance-kria.local
# or via UART 
sudo minicom -D /dev/ttyUSB1 -b 115200

If you use UART, you might need to use other port than /dev/ttyUSB1 (/dev/ttyUSB0, /dev/ttyUSB2, /dev/ttyUSB3).

Kria doesn't need password, but it requires you using "root" user.
Extra setup for the robot
Setting up 2 Ethernet cables for demo

You can connect your laptop to the Kria's built-in Ethernet port and the Unitree Go2 to the Kria through a USB-to-Ethernet adapter.

IMPORTANT : I tried the other way around and to connect my laptop through adapter, but firewall on my laptop was blocking the connection. It is possible to make it work, but I suspect that it will be a waste of time.

Connect to Kria from your laptop:

# Either via ssh
ssh root@durance-kria.local
# or via UART 
sudo minicom -D /dev/ttyUSB1 -b 115200

Then:

ip -br link

You should see something like:

lo               UNKNOWN
eth0             UP
enu1u1c2         UP

Here:

    eth0 is the Kria's built-in Ethernet port, connected to your laptop.
    enu1u1c2 is the USB Ethernet adapter, connected to Go2. Careful, depending on which of 4 USB ports you use, you might end up with different names : enu1u4c2, enu1u3c2, enu1u2c2, enu1u1c2

Normally, if I did everything correctly, you should already have en established connection to Robot because of /etc/systemd/network/10-unitree-go2.network file on Kria. You might want to modify it and restart systemd services afterwards, but probably not.

NOTE : I hardcoded MAC address in /etc/systemd/network/10-unitree-go2.network for simplicity. Make sure that your Ethernet-USB adapter has the same MAC. If not, change 10-unitree-go2.network and reboot.
Testing that connection to Robot works

The Go2's controller commonly uses 192.168.123.161, and Unitree's official ROS 2 instructions recommend configuring the Kria-side interface as 192.168.123.99/24

Ideally, by this point everything should work. Test it with :

# 1. Show both interfaces
ip -br addr

# 2. Verify the route to Go2
ip route get 192.168.123.161

# 3. Test the robot connection
ping -I enu1u4c2 -c 4 192.168.123.161

# 4. Verify the laptop connection is still working
ip route get <LAPTOP_IP>

# 5. With CycloneDDS configured:
source /opt/ros/humble/setup.bash
ros2 topic list -t

Using Kria
Making sure that Kria goes well

Run

lsusb

I expect at least 4 lines or more :

Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 0424:2744 Microchip Technology, Inc. (formerly SMSC) Hub
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 002 Device 002: ID 0424:5744 Microchip Technology, Inc. (formerly SMSC) Hub

If you only have 2, then SD card is very likely to be faulty.

If you have Ethernet cable connect, feel free to check that all IPs are set :

ifconfig

I expect at least eth0 and lo interfaces with assigned IP addresses :

eth0      Link encap:Ethernet  HWaddr 00:0A:35:24:89:FA  
          inet addr:192.168.77.1  Bcast:192.168.77.255  Mask:255.255.255.0
...
lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0

Running applications
SPLEAT server (Go server)

Normally you have a SPLEAT server (Go Server) already running :

systemctl status spleat-server.service
# or for logs :
journal -u spleat-server.service

If not, you can start it like this. This command is also useful if you want to restart it in a separate terminal to get logs directly into stdin :

systemctl stop spleat-server.service && xmutil loadapp spleat && backward-server -c /root/.config/spleat-server/config.yaml

Durance Embedded Stream processing OT

On some combinations of Kria and cameras, application can completely freeze on the 1st launch. You can avoid this by launching this workaround application 1 time after each reboot:

despot_test_camera

Application should take several seconds. If it is stuck for more than 10 seconds, then Kria board might be faulty. I tested them, so it's unlikely.

Afterwards, you can run the application like this :

# If you have no screen :
despot_send_and_forget --to_display 0
# If you have a screen connected :
DISPLAY=:0 SDL_VIDEODRIVER=x11 despot_send_and_forget --to_display 1

Cross-compilation for Unitree GO2 robot

Overall, the procedure is like this :

    you will need to Download SDK installer for your laptop (x86_64 or ARM)
    you need to build Docker image and run cross-compilation through it (because all dependencies are on this Docker image)
    you cross-compile using provided scripts or you figure out your own solution to cross-compiling
    you copy ROS2 overlay (folder of files basically) to Kria either via scp (ssh copy) or by inserting SD card into your laptop and mounting+updating it

There is more information in a separate document.
