import subprocess
import csv
import xml.etree.ElementTree as ET

def update_xml(file, params, reps):
	tree = ET.parse(file)
	root = tree.getroot()
	root[0].set('repetitions',str(reps))
	root[0][8][0].set('value', str(params[0])) #duration
	root[0][10][0].set('value', str(params[1])) #infectiousness
	root[0][11][0].set('value', str(params[2])) #transmission-radius
	root[0][12][0].set('value', str(params[3])) #chance-recover
	root[0][14][0].set('value', str(params[4])) #mobility
	root[0][15][0].set('value', str(params[5])) #immunity-duration
	with open(file, 'wb') as f:
		f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE experiments SYSTEM "behaviorspace.dtd">\n'.encode('utf8'))
		tree.write(f, 'utf-8')
		
def parse_csv(file, reps):
	f = open(file)
	data = list(csv.reader(f))
	data = data[7:]
	deaths = [data[i][10] for i in range(len(data))]
	ticks = [data[i][11] for i in range(len(data))]
	people = [data[i][12] for i in range(len(data))]
	virus = [data[i][13] for i in range(len(data))]
	deaths_avg = 0
	ticks_avg = 0
	people_avg = 0
	virus_avg = 0
	for d in deaths:
		deaths_avg += int(d)
	deaths_avg = deaths_avg / float(reps)
	for t in ticks:
		ticks_avg += int(t)
	ticks_avg = ticks_avg / float(reps)
	for p in people:
		if p == "true":
			people_avg += 1.0
	people_avg = people_avg / float(reps)
	for v in virus:
		if v == "true":
			virus_avg += 1.0
	virus_avg = virus_avg / float(reps)
	return [deaths_avg, ticks_avg/10, people_avg*10, virus_avg*10, 0., 0.];
			
def run_model(params, reps):
    for j in range(len(params)):
        if params[j] < 0:
            params[j] = 0
    for i in [0,1,3,5]:
        params[i] *= 100
    update_xml('testexp.xml', params, reps)
    subprocess.call(['rm','test.csv'])
    subprocess.call(['/Applications/NetLogo 6.1.0/netlogo-headless.sh', '--model','Virus-Parameter-Learning.nlogo', '--setup-file','testexp.xml', '--experiment', 'Test', '--table', '/Users/old/Documents/NUCS/Y3Q2/DL/Learning-to-Simulate/test.csv'])
    return parse_csv('test.csv', reps);
	

print(run_model([.2, .65, 1.5, .9, 1, .52], 1))


