#!/usr/bin/env groovy

// PROJECT CONFIGURATION
configs = [
	[
		pythonVersion: 'python3', //version de python pour créer le virtualenv
		srcDirs: ['pdcleaner'], //dossier contenant les sources python (pour le linter)
	],

	[
		node: 'compil-win64-vc14',
		testSteps: ["pytest"],
	],
	[
		node: 'compil-debian10',
		testSteps: ["pylint","flake8","bandit","pytest","import"],
		publish: true, //publication sur sonar
		package: true, //création et deploiement du tar.gz/wheel
	]
]

@Library ("utils@stages") _
def pythonStages = new com.eurodecision.jenkins.PythonStages (configs)

properties ([gitLabConnection ('edgitlab')])

pythonStages.handleCheckout()

gitlabBuilds (builds: ['build', 'test', 'publish', 'deploy']) {
   gitlabCommitStatus ('build') {
      pythonStages.build ()
   }

   gitlabCommitStatus ('test') {
      pythonStages.test ()
   }

   gitlabCommitStatus ('publish') {
      pythonStages.publish ()
   }

   gitlabCommitStatus ('deploy') {
      pythonStages.package_and_deploy ()
   }
}

pythonStages.clean()

