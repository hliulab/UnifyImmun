import subprocess
for i in range(1,2):
    print('Round {} Started:'.format(i))
    subprocess.run(["python", "HLA_1.py"])
    print('HLA_1,round{} finished'.format(i))
    subprocess.run(["python", "TCR_1.py"])

    print('TCR_1,round{} finished'.format(i))

    subprocess.run(["python", "HLA_2.py"])
    print('HLA_2,round{} finished'.format(i))

    subprocess.run(["python", "TCR_2.py"])
    print('TCR_2,round{} finished'.format(i))

