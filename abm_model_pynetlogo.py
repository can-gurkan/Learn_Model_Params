import pyNetLogo

def run_model (model, params, n):
	for i in [[0,0,2],[1,0,1],[2,0,4],[3,0,1],[4,0,2],[5,0,2]]:
		if params[i[0]] < i[1]:
			params[i[0]] = i[1]
		if params[i[0]] > i[2]:
			params[i[0]] = i[2]	
	for i in [0,1,3,5]:
		params[i] *= 100
	model.command('setup')
	model.command('set infectiousness ' + str(params[1]))
	model.command('set chance-recover ' + str(params[3]))
	model.command('set duration ' + str(params[0]))
	model.command('set transmission-radius ' + str(params[2]))
	model.command('set mobility ' + str(params[4]))
	model.command('set immunity-duration ' + str(params[5]))
	model.command('no-display')
	model.command('go')
	return model.report('run-experiment ' + str(n))

def init_model():	
	netlogo = pyNetLogo.NetLogoLink(gui=False, netlogo_home="/home/lokitekone/NetLogo 6.1.1", netlogo_version="6.1")
	netlogo.load_model('./Virus-Parameter-Learning.nlogo')
	return netlogo
	
	
