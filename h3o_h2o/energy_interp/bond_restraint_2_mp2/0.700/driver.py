import ipi_driver2 as drv
from efv import efv_scan
from sys import argv

atoms = drv.Atoms(efv_scan)
client = drv.SocketClient(unixsocket=argv[1])
client.run(atoms)
