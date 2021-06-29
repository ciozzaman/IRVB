import subprocess
import os


os.chdir('D:\\IRVB')
subprocess.call([r'D:\\IRVB\\getshotno.bat'])




from ftplib import FTP
import os
os.chdir('D:\\IRVB')
ftp = FTP('daq0.mast.l')
ftp.login('anonymous','anon')
with open('mshot.dat', 'wb') as fp:
    ftp.retrbinary('RETR mshot.dat', fp.write)
ftp.quit()

# both work
