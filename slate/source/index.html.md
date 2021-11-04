---
title: eCTF Documentation

language_tabs: # must be one of https://git.io/vQNgJ
  - python
  - c
  - make

toc_footers:
  - <a href='https://github.com/macattack99'>Developer for Documentation Pages</a>

includes:
  - errors

search: true
---

# Introduction


Welcome to Team #root’s Documentation! Below you can find our protocol and implementation of the 2017 MITRE ECTF challenge! We go through each tool, highlighting the purpose we intended it to have in the system and providing examples/explanations for how we implemented it.


# bl_build


```python
#!/usr/bin/env python
import os
import random
import shutil
import subprocess
import sys
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from intelhex import IntelHex
from base64 import b64encode
import json


FILE_DIR = os.path.abspath(os.path.dirname(__file__))

def make_bootloader(readback_password=None,firmware_key=None,mac_key=None):
    """
    Build the bootloader from source.
    Return:
        True if successful, False otherwise.
    """
    # Change into directory containing bootloader.
    os.chdir('../bootloader')

    subprocess.call('make clean', shell=True)
    # Call make in subprocess to build bootloader.
    if readback_password is not None or firmware_key is not None:
        status = subprocess.call('make -f Makefile READBACK_PASSWORD="%s" FIRMWARE_KEY="%s" MAC_KEY="%s"' % (readback_password,firmware_key,mac_key), shell=True)
    else:
        status = subprocess.call('make -f Makefile')


    # Return True if make returned 0, otherwise return False.
    return (status == 0)

def copy_artifacts():
    """
    Copy bootloader build artifacts into the host tools directory.
    """
    # Get directory containing this file (host_tools).
    dst_dir = FILE_DIR

    # Get directory containing bootloader output (bootloader).
    src_dir = os.path.join(os.path.dirname(dst_dir), 'bootloader')

    # Copy build artifacts from bootloader directory.
    shutil.copy(os.path.join(src_dir, 'flash.hex'), dst_dir)
    shutil.copy(os.path.join(src_dir, 'eeprom.hex'), dst_dir)
    shutil.copy(os.path.join(dst_dir, 'lfuse.hex'), src_dir)
    shutil.copy(os.path.join(dst_dir, 'hfuse.hex'), src_dir)
    shutil.copy(os.path.join(dst_dir, 'efuse.hex'), src_dir)

def generate_readback_password():
    """
    Generate secret password for readback tool and store to secret file.
    """
    # Generate AES key for use in readback password
    readback_password = get_random_bytes(32)

    return readback_password

def generate_firmware_specifics():
    #Generate AES key for encrypting firmware
    firmware_key = get_random_bytes(32)
    mac_key = get_random_bytes(32)
    return (firmware_key,mac_key)

def format_key(key):
    #change keys into a 0x00,0x00,etc.. format that the bootloader can understand
    key = bytes(key).encode('hex')
    a = list(map(''.join, zip(*[iter(key)]*2)))
    key = ''.join(map(lambda i: '0x' + i + ',', a))
    key = '{' + key + '}'
    key = key[0:len(key)-2]
    key = key + '}'
    return(key)


def write_fuse_file(fuse_name, fuse_value):
    hex_file = IntelHex()
    hex_file[0] = fuse_value

    with open(os.path.join(FILE_DIR, fuse_name + '.hex'), 'wb+') as outfile:
        hex_file.tofile(outfile, format='hex')

if __name__ == '__main__':
    readback_password = generate_readback_password()
    firmware_key,mac_key = generate_firmware_specifics()

    #export to secret_build_output
    json_k = ['readback_password','firmware_key','mac_key']
    json_v = [b64encode(x).decode('utf-8') for x in firmware_key,readback_password,mac_key]
    result = json.dumps(dict(zip(json_k,json_v)))
    with open('secret_build_output.txt','w') as secret_build_output:
        secret_build_output.write(result)

    # Format keys to be sent to bootloader
    readback_password = format_key(readback_password)
    firmware_key = format_key(firmware_key)
    mac_key = format_key(mac_key)

    #build the bootloader from source using information generated above
    if not make_bootloader(readback_password=readback_password,firmware_key=firmware_key,mac_key = mac_key):
        print "ERROR: Failed to compile bootloader."
        sys.exit(1)
    write_fuse_file('lfuse', 0xFF)
    write_fuse_file('hfuse', 0x18)
    write_fuse_file('efuse', 0xFC)
    copy_artifacts()
```

This tool is responsible for building the bootloader from source and copying the build outputs into the host tools directory for programming.

Generates a AES-128 key, readback password, and mac key → All of which get converted to hex so that it can be understood by the bootloader

  Important to use fast cipher (like AES-CBC) due to the limited amount of memory

  These get added as #define statements in the bootloader

AES key, mac key, and Readback password are stored in secret_build_output.txt

  Also create ASCII in Secret_build_output.txt

Generate a seed for a pseudo-random number generator (for use during readback)

# bl_configure

```python
#!/usr/bin/env python
import argparse
import os
import serial
import shutil

FILE_PATH = os.path.abspath(__file__)

def generate_secret_file():
    """
    Compile all secrets from build and configuration and store to secret file.
    """
    # Get directory containing this file (host_tools).
    directory = os.path.dirname(FILE_PATH)

    # Copy secret build output to secret configure output.
    shutil.copyfile(os.path.join(directory, 'secret_build_output.txt'),
                    os.path.join(directory, 'secret_configure_output.txt'))

    # If there were additional secret parameters to output, the file could be
    # edited here.

def configure_bootloader(serial_port):
    """
    Configure bootloader using serial connection.
    """
    # If there were online configuration or checking of the bootloader using
    # the serial port it could be added here.
    pass


if __name__ == '__main__':
    # Argument parser setup.
    parser = argparse.ArgumentParser(description='Bootloader Config Tool')
    parser.add_argument('--port', help='Serial port to use for configuration.',
                        required=True)
    args = parser.parse_args()

    # Create serial connection using specified port.
    serial_port = serial.Serial(args.port)

    # Do configuration and then close port.
    try:
        configure_bootloader(serial_port)
    finally:
        serial_port.close()

    # Generate secret file.
    generate_secret_file()
```

This tool makes sure that bl_build was run successfully

It communicates with the device through serial communication and sends a message to the bootloader to take it out of the mode that generates bl_build and into it's normal state.

All values stored in secret_build_output.txt are also transferred to secret_configure_output.txt

# fw_protect

```python
#!/usr/bin/env python
import argparse
import shutil
import struct
import ast

from cStringIO import StringIO
from intelhex import IntelHex
from base64 import b64decode
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

def format_key(key):
    #change keys into a 0x00,0x00,etc.. format that the bootloader can understand
    key = bytes(key).encode('hex')
    a = list(map(''.join, zip(*[iter(key)]*2)))
    key = ''.join(map(lambda i: '0x' + i + ',', a))
    key = '{' + key + '}'
    key = key[0:len(key)-2]
    key = key + '}'
    return(key)

if __name__ == '__main__':

    #import secret_configure_output file as a dictionary
    with open('secret_build_output.txt','r') as secret_configure_output:
        keyDict = ast.literal_eval(secret_configure_output.read())

    parser = argparse.ArgumentParser(description='Firmware Update Tool')

    parser.add_argument("--infile",
                        help="Path to the firmware image to protect.",
                                                required=True)
    parser.add_argument("--outfile", help="Filename for the output firmware.",
                        required=True)
    parser.add_argument("--version", help="Version number of this firmware.",
                        required=True)
    parser.add_argument("--message", help="Release message for this firmware.",
                        required=True)

    args = parser.parse_args()

    # Parse Intel hex file.
    firmware = IntelHex(args.infile)

    # Get version and size.
    firmware_size = firmware.maxaddr() + 1
    version = int(args.version)

    # Add release message to end of hex (null-terminated).
    sio = StringIO()
    firmware.putsz(firmware_size, (args.message + '\0'))
    firmware.write_hex_file(sio)
    hex_data = sio.getvalue()

    #encrypt hex file
    cipher = AES.new(b64decode(keyDict['firmware_key']), AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(hex_data, AES.block_size))
    # Format version and firmware size
    metadata = struct.pack('>HH',version,firmware_size)

    '''
        #compute cbc mac of the data
        package = metadata + ciphertext
        iv = bytearray(16)
        macCipher = AES.new(b64decode(keyDict['mac_key']), AES.MODE_CBC, iv=iv)
        macText = cipher.encrypt(pad(package, AES.block_size))
        mac = macText[-16:]
    '''
    with open(args.outfile, 'wb+') as outfile:
        hexOutput = IntelHex()
        hexOutput.putsz(0,ciphertext)
        hexOutput.write_hex_file(sio)
        finalOutput = sio.getvalue()
        finalOutput = metadata + '\n' + cipher.iv + '\n' +  finalOutput
        outfile.write(finalOutput)
```
> 

* Stores unprotected firmware and accesses the nonce/key in Secret_build_output.txt

* Encrypts the firmware using AES-CBC and then creates  CBC-MAC by encrypting the firmware a second time with the mac key from bl_build and using the last 16 bytes as the MAC (done to that the MAC is not just the last 16 bytes of the encrypted firmware).

* We have a static, default IV so that it is known so it can be decrypted as long as you know the key

* Exports encrypted firmware and the CBC-MAC to be used by the fw_update tool

* Firmware package: Version + firmware size + encrypted firmware + CBC-MAC

* Sends firmware package to fw_update

# fw_update

```python
#!/usr/bin/env python
import argparse
import json
import os
import serial
import struct
import sys
import time

from cStringIO import StringIO
from intelhex import IntelHex

RESP_OK = b'\x00'


class Firmware(object):
    """
    Helper for making frames.
    """

    BLOCK_SIZE = 16

    def __init__(self, fw_filename):
        with open(fw_filename, 'rb') as fw_file:
            self.metadata = fw_file.readline()[:-1]
            self.iv = fw_file.readline()[:-1]
            self.hex_data = StringIO(fw_file.read())


        self.reader = IntelHex(self.hex_data)

    def frames(self):
        # The address is not sent, so we currently only support a single segment
        # starting at address 0.
        if len(self.reader.segments()) > 1:
            raise RuntimeError("ERROR: Hex file contains multiple segments.")

        for segment_start, segment_end in self.reader.segments():

            if segment_start != 0:
                raise RuntimeError("ERROR: Segment in Hex file does not start at address 0.")

            # Construct frame from data and length.
            for address in range(segment_start, segment_end, self.BLOCK_SIZE):

                # Frame should be BLOCK_SIZE unless it is the last frame.
                if address + self.BLOCK_SIZE <= segment_end:
                    data = self.reader.tobinstr(start=address,
                                                size=self.BLOCK_SIZE)
                else:
                    data = self.reader.tobinstr(start=address,
                                                size=segment_end - address)
                # Get length of frame.
                length = len(data)
                frame_fmt = '>H{}s'.format(length)

                # Construct frame.
                yield struct.pack(frame_fmt, length, data)

    def close(self):
        self.reader.close()


class NMConn:
    def __init__(self):
        self.rx = os.open('/tmp/uart1_tx', os.O_RDONLY)
        time.sleep(1)
        self.tx = os.open('/tmp/uart1_rx', os.O_WRONLY)

    def read(self, n=1):
        return os.read(self.rx, n)

    def write(self, msg):
        os.write(self.tx, msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Firmware Update Tool')

    parser.add_argument("--port", help="Serial port to send update over.",
                        required=True)
    parser.add_argument("--firmware", help="Path to firmware image to load.",
                        required=True)
    parser.add_argument("--debug", help="Enable debugging messages.",
                        action='store_true')
    parser.add_argument("--test", help="Connec to named pipe for testing.",
                        action='store_true')
    args = parser.parse_args()

    # Open serial port. Set baudrate to 115200. Set timeout to 2 seconds.
    print('Opening serial port...')

    if args.test:
        ser = NMConn()

    else:
        ser = serial.Serial(args.port, baudrate=115200, timeout=2)


    # Open our firmware file.
    print('Opening firmware file...')
    firmware = Firmware(args.firmware)

    print('Waiting for bootloader to enter update mode...')

    while ser.read(1) != 'U':
        pass


    # Send size and version to bootloader.
    if args.debug:
        print(firmware.metadata.encode('hex'))
    ser.write(firmware.metadata)

    # Wait for an OK from the bootloader.
    resp = ser.read()
    if resp != RESP_OK:
        raise RuntimeError("ERROR: Bootloader responded with {}".format(repr(resp)))


    #write iv into memory
    print("writing iv:" + firmware.iv.encode('hex'))
    ser.write(firmware.iv)

    # Wait for an OK from the bootloader.
    resp = ser.read()
    if resp != RESP_OK:
        raise RuntimeError("ERROR: Bootloader responded with {}".format(repr(resp)))

    for idx, frame in enumerate(firmware.frames()):
        if args.debug:
            print("Writing frame {} ({} bytes)...".format(idx, len(frame)))

        ser.write(frame)  # Write the frame...

        if args.debug:
            print(frame.encode('hex'))

        resp = ser.read()  # Wait for an OK from the bootloader

        time.sleep(0.1)

        if resp != RESP_OK:
            raise RuntimeError("ERROR: Bootloader responded with {}".format(repr(resp)))

        if args.debug:
            print("Resp: {}".format(ord(resp)))

    print("Done writing firmware.")



    # Send a zero length payload to tell the bootlader to finish writing
    # it's page.
    ser.write(struct.pack('>H', 0x0000))
```

> Firmware Updater Tool

> A frame consists of two sections:
> 1. Two bytes for the length of the data section
> 2. A data section of length defined in the length section

> In our case, the data is from one line of the Intel Hex formated .hex file
> We write a frame to the bootloader, then wait for it to respond with an
> OK message so we can write the next frame. The OK message in this case is
> just a zero

Sends the encrypted firmware and the plaintext nonces to the bootloader through UART

* Universal Asynchronous Receiver Transmitter module - International Journal of Scientific Engineering and Technology Research 
Clock generator circuit provides clock frequencies for clock modules 

* The bytes of data is stored in 128 bit shift register to be encrypted with AES and shifted to the input of the UART transmitter

* The bits are assembled by the UART receiver

# readback

```python
#!/usr/bin/env python
"""
Memory Readback Tool
A frame consists of four sections:
1. One byte for the length of the password.
2. The variable-length password.
3. Four bytes for the start address.
4. Four bytes for the number of bytes to read.
  [ 0x01 ]  [ variable ]  [ 0x04 ]    [ 0x04 ]
-------------------------------------------------
| PW Length | Password | Start Addr | Num Bytes |
-------------------------------------------------
"""

from os import urandom

import serial
import struct
import sys
import argparse
from Crypto.Cipher import AES
from base64 import b64decode
import ast
from Crypto.Util import Padding


RESP_OK = b'\x00'
RESP_ERROR = b'\x01'

def construct_request(start_addr, num_bytes):
    # Read in secret password from file.
    with open('secret_build_output.txt', 'rb') as secret_file:
        keyDict = ast.literal_eval(secret_file.read())
    cipher = AES.new(keyDict["readback_password"],AES.MODE_CBC)
    packet = struct.pack('>II', start_addr, num_bytes)
    ciphertext = cipher.encrypt(packet)
    # TODO: check MAC implementation
    tag = ciphertext[-16:]

    return (tag + cipher.iv + ciphertext)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Memory Readback Tool')

    parser.add_argument("--port", help="Serial port to send update over.",
                        required=True)
    parser.add_argument("--address", help="First address to read from.",
                        required=True)
    parser.add_argument("--num-bytes", help="Number of bytes to read.",
                        required=True)
    parser.add_argument("--datafile", help="File to write data to (optional).")

    args = parser.parse_args()

    request = construct_request(int(args.address), int(args.num_bytes))

    # Open serial port. Set baudrate to 115200. Set timeout to 2 seconds.
    ser = serial.Serial(args.port, baudrate=115200, timeout=2)

    # Wait for bootloader to reset/enter readback mode.
    while ser.read(1) != 'R':
        pass

    # Send the request to start communication with bootloader
    ser.write(request)

    # Read the data and write it to stdout (hex encoded).
    #TODO: DECRYPT INFO COMING FROM BOOTLOADER

    # Gets the random 16 byte nonce from bootloader
    bootnonce = ser.read(16)

    # Creates the random 16 byte nonce in the readback tool
    toolnonce = urandom(16)

    with open('secret_build_output.txt', 'rb') as secret_file:
        keyDict = ast.literal_eval(secret_file.read())
    cipher2 = AES.new(keyDict["readback_password"],AES.MODE_CBC, iv=toolnonce)

    # Encrypts boot nonce with tool nonce and secret password
    encbootnonce = cipher2.encrypt(bootnonce)

    # Sends tool nonce and encrypted boot nonce to bootloader
    ser.write(toolnonce)
    ser.write(encbootnonce)

    data = ser.read(int(args.num_bytes))
    print(data.encode('hex'))

    # Write raw data to file (optional).
    if args.datafile:
        with open(args.datafile, 'wb+') as datafile:
            datafile.write(data)
```


> 

* Imports modules 

* Sets the proper response in hex 

* Reads in the secret password from file 

  Sets cipher from AES key 

  Encrypts the packet 

  Sets MAC tag as last 16 bytes of the encrypted packet 

  Returns tag, cipher iv, and the encrypted packet 

* Main function of readback tool: 

  Opens ports and baud rate to enable communication with bootloader 

  Sends request to begin communication with bootloader 

  Takes readback master password (AES-128 key)

  Readback tool initiates communication with bootloader 

  Bootloader sends back 16 byte string 

  Readback generates random nonce

  Readback must use AES-128 to encrypt the string using the nonce and sends back the encrypted text along with the nonce (test)
  The bootloader checks the encryption and, if authenticated, returns the readback data

  Encrypts data and sends any technician commands with nonces.

  Whats the print data and optional write raw data to file


# Bootloader

```c
#ifdef PC
#define EEMEM
#include "e_eeprom.h"
#include "e_io.h"
#include "e_wdt.h"
#include "e_pgmspace.h"
#include <unistd.h>

#else
#include <avr/io.h>
#include <avr/boot.h>
#include <avr/wdt.h>
#include <avr/interrupt.h>
#include <avr/pgmspace.h>

#include <util/delay.h>
#endif

#include <string.h>
#include "aes.h"
#include "pin.h"
#include <stdint.h>
#include <stdio.h>

#include "uart.h"
#include "flash.h"



#define OK    ((unsigned char)0x00)
#define ERROR ((unsigned char)0x01)



void load_firmware(void);
void boot_firmware(void);
void readback(void);
void cbc_encrypt(aes256_ctx_t *ctx, uint8_t *iv, uint8_t *data, int len);
void cbc_decrypt(aes256_ctx_t *ctx, uint8_t *iv, uint8_t *data, int len);
void cbc_mac(aes256_ctx_t* ctx, uint8_t* data, uint8_t* mac, int len);
uint8_t mac_verify(uint8_t* mac_key, uint8_t* mac, uint8_t* data);
void cbc_decrypt_firmware(aes256_ctx_t *ctx, uint8_t *pBlock, uint8_t *data, int len);
int cst_time_memcmp_safest1(const void *m1, const void *m2, size_t n);


uint16_t fw_size EEMEM = 0;
uint16_t fw_version EEMEM = 0;

const unsigned char FW_KEY[32] = FIRMWARE_KEY;
const unsigned char RB_PASS[32] = READBACK_PASSWORD;
const unsigned char M_KEY[32] = MAC_KEY;

int main(void)
{

    // Init UART1 (virtual com port)
    UART1_init();

    UART0_init();
    wdt_reset();
#ifdef PC
    PIN_init();

    // If jumper is present on pin 2, load new firmware.
    if(!PIN_update())
    {
        UART1_putchar('U');
        load_firmware();
    }
    else if(!PIN_readback())
    {
        UART1_putchar('R');
        readback();
    }
    else
    {
      UART1_putchar('B');
      boot_firmware();

    }
#else
    DDRB &= ~((1 << PB2) | (1 << PB3));

    // Enable pullups - give port time to settle.
    PORTB |= (1 << PB2) | (1 << PB3);

    // If jumper is present on pin 2, load new firmware.
    if(!(PINB & (1 << PB2)))
    {
    UART1_putchar('U');
    load_firmware();
    }
    else if(!(PINB & (1 << PB3)))
    {
    UART1_putchar('R');
    readback();
    }
    else
    {
    UART1_putchar('B');
    boot_firmware();
    }
#endif
} // main

/*
 * Interface with host readback tool.
 */
void readback(void)
{
    // Start the Watchdog Timer
    wdt_enable(WDTO_2S);

    // Read in start address (4 bytes).
    uint32_t start_addr = ((uint32_t)UART1_getchar()) << 24;
    start_addr |= ((uint32_t)UART1_getchar()) << 16;
    start_addr |= ((uint32_t)UART1_getchar()) << 8;
    start_addr |= ((uint32_t)UART1_getchar());

    wdt_reset();

    // Read in size (4 bytes).
    uint32_t size = ((uint32_t)UART1_getchar()) << 24;
    size |= ((uint32_t)UART1_getchar()) << 16;
    size |= ((uint32_t)UART1_getchar()) << 8;
    size |= ((uint32_t)UART1_getchar());

    wdt_reset();

    // Read the memory out to UART1.
    for(uint32_t addr = start_addr; addr < start_addr + size; ++addr)
    {
        // Read a byte from flash.
        unsigned char byte = pgm_read_byte_far(addr);
        wdt_reset();

        // Write the byte to UART1.
        UART1_putchar(byte);
        wdt_reset();
    }

    while(1) __asm__ __volatile__(""); // Wait for watchdog timer to reset.
}


/*
 * Load the firmware into flash.
 */
void load_firmware(void)
{
    uint8_t pageBuffer[SPM_PAGESIZE];
    uint8_t previousBlock[16];
    uint8_t tempBlock[16];
    int frame_length = 0;
    unsigned char rcv = 0;
    unsigned char data[SPM_PAGESIZE]; // SPM_PAGESIZE is the size of a page.
    unsigned int data_index = 0;
    unsigned int page = 0;
    uint16_t version = 0;
    uint16_t size = 0;
    uint8_t fw_iv[16];
    aes256_ctx_t ctx;

    aes256_init(FW_KEY, &ctx);

    // Start the Watchdog Timer
    wdt_enable(WDTO_2S);

    /* Wait for data */
    while(!UART1_data_available())
    {
        __asm__ __volatile__("");
    }

    // Get version.
    rcv = UART1_getchar();
    version = (uint16_t)rcv << 8;
    rcv = UART1_getchar();
    version |= (uint16_t)rcv;

    // Get size.
    rcv = UART1_getchar();
    size = (uint16_t)rcv << 8;
    rcv = UART1_getchar();
    size |= (uint16_t)rcv;

    // Compare to old version and abort if older (note special case for version
    // 0).
    if (version != 0 && version < eeprom_read_word(&fw_version))
    {
        UART1_putchar(ERROR); // Reject the metadata.
        // Wait for watchdog timer to reset.
        while(1)
        {
            __asm__ __volatile__("");
        }
    }
    else if(version != 0)
    {
        // Update version number in EEPROM.
        wdt_reset();
        eeprom_update_word(&fw_version, version);
    }

    // Write new firmware size to EEPROM.
    wdt_reset();
    eeprom_update_word(&fw_size, size);
    wdt_reset();

    UART1_putchar(OK); // Acknowledge the metadata.

    //program the iv into memory
    for(int i = 0; i < 16; i++){
        wdt_reset();
        fw_iv[i] = UART1_getchar();
      }

    //set previousblock to the value of iv
    memcpy(previousBlock,fw_iv,16);

    wdt_reset();
    UART1_putchar(OK); // Acknowledge the iv

    /* Loop here until you can get all your characters and stuff */
    while(1)
    {
        wdt_reset();
        // Get two bytes for the length.
        rcv = UART1_getchar();
        frame_length = (int)rcv << 8;
        rcv = UART1_getchar();
        frame_length += (int)rcv;

        UART0_putchar((unsigned char)rcv);
        wdt_reset();

        // Get the number of bytes specified
        for(int i = 0; i < frame_length; ++i){
            wdt_reset();
            data[data_index] = UART1_getchar();
            data_index += 1;
        } //for

        // If we filed our page buffer, program it
        if(data_index == SPM_PAGESIZE || frame_length == 0)
        {
            wdt_reset();

            //remeber the value of the ciphertext
            for(int i=0;i<SPM_PAGESIZE;i=i+16){
              wdt_reset();
              memcpy(tempBlock,data+i,16);

              //decrypt firmware
              cbc_decrypt_firmware(&ctx,previousBlock,tempBlock,16);
              memcpy(previousBlock,data+i,16);
              memcpy(data+i,tempBlock,16);
            }
            //flash the decrypted firmware
            //hexprint(data,SPM_PAGESIZE);
            program_flash(page, data);
            page += SPM_PAGESIZE;
            data_index = 0;

#if 1
            // Write debugging messages to UART0.
            UART0_putchar('P');
            UART0_putchar(page>>8);
            UART0_putchar(page);
#endif
            wdt_reset();
        } // if
      UART1_putchar(OK); // Acknowledge the frame.
    } // while(1)
  }

/*
 * Ensure the firmware is loaded correctly and boot it up.
 */
void boot_firmware(void)
{
    // Start the Watchdog Timer.
    wdt_enable(WDTO_2S);

    // Write out the release message.
    uint8_t cur_byte;
    uint8_t test; //test
    uint32_t addr = (uint32_t)eeprom_read_word(&fw_size);

    // Reset if firmware size is 0 (indicates no firmware is loaded).
    if(addr == 0)
    {
        // Wait for watchdog timer to reset.
        while(1) __asm__ __volatile__("");
    }

    wdt_reset();

    //test
    test = pgm_read_byte_far(0000);
    UART0_putchar(test&0xff);
    test = pgm_read_byte_far(0001);
    UART0_putchar(test&0xff);

    // Write out release message to UART0.
    do
    {
        wdt_reset();
        cur_byte = pgm_read_byte_far(addr);
        UART0_putchar(cur_byte);
        ++addr;
    } while (cur_byte != 0);

    // Stop the Watchdog Timer.
    wdt_reset();
    wdt_disable();

    /* Make the leap of faith. */
    asm ("jmp 0000");
}

void cbc_encrypt(aes256_ctx_t *ctx, uint8_t *iv, uint8_t *data, int len){
    unsigned char *buf, *ivbuf;
    ivbuf = iv;
    buf = data;
    while(len>0){
        //xor plaintext buffer with iv buffer
        for(int i =0; i < 16; ++i){
            buf[i] ^= ivbuf[i];
        }
        //encrypts the xored block
        aes256_enc(buf,ctx);
        //save ciphertext block for next iteration
        memcpy(ivbuf, buf, 16);
        //loops
        buf += 16;
        len -= 16;
    }
}

void cbc_decrypt_firmware(aes256_ctx_t *ctx, uint8_t *pBlock, uint8_t *data, int len){
  unsigned char *buf, *blockBuf;
  blockBuf = pBlock;
  buf = data;
  unsigned char tmp[16];
  //save ciphertext block
  memcpy(tmp, buf, 16);
  //decrypt ciphertext block
  aes256_dec(buf, ctx);
  //xor ciphertext with previous block/iv
  for(int i =0; i < 16; ++i){
      buf[i] ^= blockBuf[i];
    }
}

void cbc_decrypt(aes256_ctx_t *ctx, uint8_t *iv, uint8_t *data, int len){
    unsigned char *buf, *ivbuf;
    ivbuf = iv;
    buf = data;
    while(len>0){
        unsigned char tmp[16];
        //save ciphertext block
        memcpy(tmp, buf, 16);
        //decrypt ciphertext block
        aes256_dec(buf, ctx);
        //xor ciphertext with previous block/iv
        for(int i =0; i < 16; ++i){
            buf[i] ^= ivbuf[i];
        }
        memcpy(ivbuf, tmp, 16);
        //loops
        buf += 16;
        len -= 16;
    }
}

void cbc_mac(aes256_ctx_t* ctx, uint8_t* data, uint8_t* mac, int len){
  uint8_t iv[16];
  memset(iv,0,16);
  memcpy(mac,data,16);

  while (len > 0){
    //xor ciphertext with previous block/iv
    for(int i = 0; i < 16;i++){
        mac[i] ^= iv[i];
    }

    aes256_enc(mac,ctx);
    memcpy(iv,mac,16);

    //rinse and repeat
    len -= 16;
    data += 16;
  }
}

uint8_t mac_verify(uint8_t* mac_key, uint8_t* mac, uint8_t* data){
  aes256_ctx_t ctx;
  aes256_init(mac_key, &ctx);
  int len = sizeof(*data);
  uint8_t test_mac[16];
  memset (test_mac,0,16);
  cbc_mac(&ctx,data,test_mac,len);
  uint8_t n = cst_time_memcmp_safest1(test_mac,mac,16); //if memcmp isn't constant time we should add something
  return n; //returns 0 if true, anything else if false
}

int cst_time_memcmp_safest1(const void *m1, const void *m2, size_t n)
{
    const unsigned char *pm1 = (const unsigned char*)m1;
    const unsigned char *pm2 = (const unsigned char*)m2;
    int res = 0, diff;
    if (n > 0) {
        do {
            --n;
            diff = pm1[n] - pm2[n];
            res = (res & (((diff - 1) & ~diff) >> 8)) | diff;
        } while (n != 0);
    }
    return ((res - 1) >> 8) + (res >> 8) + 1;
}

void hexprint(uint8_t *addr, uint32_t len){ //TESTING
    for (int i = 0; i < len; i++){
        printf("%02x", addr[i]);
    }
    printf("\n");
}

```

This bootloader supports firmware up to 30kb (rejects if larger) along with 4 bytes of additional information (2 bytes for size of firmware and 2 bytes for version) that is concatenated to the beginning of the firmware in fw_protect. This also supports a message of undefined length concatenated at the end of the firmware (again in fw_protect)

This writes the firmware image to flash memory and ... (Read Jake's README... idk if we did any of that or not)

The bootloader  includes a compare function that confirms the validity of the firmware, making sure that it is a newer version (by comparing to the firmware currently installed on chip) and not invalid firmware. The version check also has an exception created for version 0.

Once these checks are passed, the bootloader retrieves the firmware pachage from UART1, along with the number of specified bytes

The bootloader then stores the iv (from fw_update) and the cipher mac (stored in secret_configure_output.txt) into memory so that the encrypted firmware can be decrypted

The firmware is then encrypted using the stored iv and key to check the mac and the unencrypted firmware is flashed (it is decrypted before being flashed)

The bootloader also contains a verification that the mac matches the one stored in the bootloader, and if it doesn't then it deletes the installed firmware and sends ERROR message back to fw_update, telling it that there was a problem verifying the mac

The release message is then written out to UART0

IDK IF I SHOULD TALK ABOUT READBACK

# Makefile

```make
# Hardware configuration settings.
MCU = atmega1284p
F_CPU = 20000000
BAUD = 115200

# Secret password default value.
READBACK_PASSWORD ?= readback_password
FIRMWARE_KEY ?= firmware_key
MAC_KEY ?= mac_key

# Tool aliases.
CC = avr-gcc
STRIP  = avr-strip
OBJCOPY = avr-objcopy
PROGRAMMER = dragon_jtag

# Compiler configurations.
# Description of CDEFS options
# -g3 -- turns on  the highest level of debug symbols.
# -ggdb3 -- turns on the highest level of debug symbols for the gdb debugger.
#
#  NOTE: The debug options shoud only affect the .elf file. Any debug symbols are stripped
#  from the .hex file so no debug info is actually loaded on the AVR. This means that removing
#  debug symbols should not affect the size of the firmware.
CDEFS = -g3 -ggdb3 -mmcu=${MCU} -DF_CPU=${F_CPU} -DBAUD=${BAUD} -DREADBACK_PASSWORD=${READBACK_PASSWORD} -DFIRMWARE_KEY=${FIRMWARE_KEY} -DMAC_KEY=${MAC_KEY}

# Description of CLINKER options:
#     -Wl,--section-start=.text=0x1E000 -- Offsets the code to the start of the bootloader section
#     -Wl,-Map,bootloader.map -- Created an additional file that lists the locations in memory of all functions.
CLINKER = -nostartfiles -Wl,--section-start=.text=0x1E000 -Wl,-Map,bootloader.map

CWARN =  -Wall
COPT = -std=gnu99 -Os -fno-tree-scev-cprop -mcall-prologues \
       -fno-inline-small-functions -fsigned-char

CFLAGS  = $(CDEFS) $(CLINKER) $(CWARN) $(COPT)

# Include file paths.
INCLUDES = -I./include -I ./crypto

SOURCES = $(wildcard src/*c)
OBJECTS = $(patsubst %.c, %.o, %(SOURCES))

CRYPTO_SRCS = $(wildcard crypto/*.c)
CRYPTO_OBJS = $(patsubst %.c, %.o, $(CRYPTO_SRCS))
CRYPTO_OBJS_x86 = $(patsubst %.c, %_x86.o, $(CRYPTO_SRCS))
# Run clean even when all files have been removed.
.PHONY: clean

all:    flash.hex eeprom.hex
	@echo  Simple bootloader has been compiled and packaged as intel hex.

pin.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c src/pin.c

flash.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c src/flash.c
$(CRYPTO_OBJS): crypto/%.o : crypto/%.c
		$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CRYPTO_OBJS_x86): crypto/%_x86.o : crypto/%.c
		gcc  $(INCLUDES) -DLOCAL  -c $< -o $@

main.o:
		gcc  $(INCLUDES) -DLOCAL  -c test/main.c -o test/main.o

#$(OBJECTS): src/%.o : src/%.change
#    $(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

bootloader_dbg.elf: $(CRYPTO_OBJS)
		$(CC) $(CFLAGS) $(INCLUDES) -o bootloader_dbg.elf
				uart.o bootloader.o sys_startup.o $(CRYPTO_OBJS)

local: $(CRYPTO_OBJS_x86) main.o
		gcc $(INCLUDES) -o local $(CRYPTO_OBJS_x86) test/main.o

uart.o:
		$(CC) $(CFLAGS) $(INCLUDES) -c src/uart.c

sys_startup.o:
		$(CC) $(CFLAGS) $(INCLUDES) -c src/sys_startup.c

bootloader.o:
		$(CC) $(CFLAGS) $(INCLUDES) -c src/bootloader.c

bootloader_dbg.elf: pin.o flash.o uart.o sys_startup.o bootloader.o $(CRYPTO_OBJS)
        # Create an .elf file for the bootloader with all debug symbols included.
	$(CC) $(CFLAGS) $(INCLUDES) -o bootloader_dbg.elf pin.o flash.o uart.o sys_startup.o bootloader.o  $(CRYPTO_OBJS)

strip: bootloader_dbg.elf
    # Create a version of the bootloder .elf file with all the debug symbols stripped.
		$(STRIP) bootloader_dbg.elf -o bootloader.elf

flash.hex: strip
		$(OBJCOPY) -R .eeprom -O ihex bootloader.elf flash.hex
		avr-size flash.hex

eeprom.hex: strip
		$(OBJCOPY) -j .eeprom --set-section-flags=.eeprom="alloc,load" --change-section-lma .eeprom=0 -O ihex bootloader.elf eeprom.hex

flash: flash.hex eeprom.hex
		avrdude -v -V -F -P usb -p m1284p -c $(PROGRAMMER)  -u -U flash:w:flash.hex:i \
                            -U eeprom:w:eeprom.hex:i \
                            -U lfuse:w:lfuse.hex:i \
                            -U hfuse:w:hfuse.hex:i \
                            -U efuse:w:efuse.hex:i

debug: flash.hex eeprom.hex
    # Launch avarice: a tool that creates a debug server for the AVR and Dragon
    Sudo avarice -R -g :4242 &
    # Launch the avr debugger avr-gdb. The configuation for this tool is included
    # in .gdbinit
		avr-gdb

clean:
		$(RM) -v *.hex *.o *.elf $(MAIN) test/main.o crypto/*.o
```

Sets default values for the Readback Password, Firmware Key, and MAC Key

Runs eeprom.hex and flash.hex
  