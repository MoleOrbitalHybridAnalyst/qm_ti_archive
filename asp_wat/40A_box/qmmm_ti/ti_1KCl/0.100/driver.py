import ipi_driver2 as drv
from qmmm_efv import efv_scan

atoms = drv.Atoms(efv_scan)
client = drv.SocketClient(unixsocket='aspqm0.100')
client.run(atoms)
