# Cytomine Python client

> Cytomine-python-client is an open-source Cytomine client written in Python. This client is a Python wrapper around Cytomine REST API gateway.

[![Build Status](https://travis-ci.com/Cytomine-ULiege/Cytomine-python-client.svg?branch=master)](https://travis-ci.com/Cytomine-ULiege/Cytomine-python-client)
[![GitHub release](https://img.shields.io/github/release/Cytomine-ULiege/Cytomine-python-client.svg)](https://github.com/Cytomine-ULiege/Cytomine-python-client/releases)
[![GitHub](https://img.shields.io/github/license/Cytomine-ULiege/Cytomine-python-client.svg)](https://github.com/Cytomine-ULiege/Cytomine-python-client/blob/master/LICENSE)

## Overview

The main access point to Cytomine data is its REST API. This client is a Python package that can be imported in an application and allows to import/export data from Cytomine-Core and Cytomine-IMS using RESTful web services e.g. to generate annotation (spatial) statistics, create regions of interest (e.g. tumor masks), add metadata to images/annotations, apply algorithms on image tiles, ...

See [documentation](http://doc.cytomine.be/display/ALGODOC/%5BDOC%5D+Data+access) for more details.

## Requirements
* Python 2.7 | 3.5+

## Install

**To install *official* release of Cytomine-python-client, see @cytomine. Follow this guide to install forked version by ULiege.** 

### Manual installation
To download and install manually the package, see [manual installation procedure](http://doc.cytomine.be/display/ALGODOC/Data+access+using+Python+client#DataaccessusingPythonclient-Installation).

### Automatic installation
To retrieve package using `pip`:

    curl -s https://packagecloud.io/install/repositories/cytomine-uliege/Cytomine-python-client/script.python.sh | bash
    pip install cytomine-python-client

See [package repository](https://packagecloud.io/cytomine-uliege/Cytomine-python-client) for details.

### In a Docker container
To ease developpement of new Cytomine software, the Cytomine-python-client package is available in Docker containers:
* [cytomineuliege/software-python3-base](https://hub.docker.com/r/cytomineuliege/software-python3-base/) provides a Python 3.5 environment with client already installed.
* [cytomineuliege/software-python2-base](https://hub.docker.com/r/cytomineuliege/software-python2-base/) provides a Python 2.7 environment with client already installed.

These Docker images are tagged with the Python client version number. Two image variants are given for each client version:
* `cytomineuliege/software-pythonX-base:<version>` is the defacto image. If you are unsure about what your needs are, you probably want to use this one.
* `cytomineuliege/software-pythonX-base:<version>-slim` is an image that does not contain all the common package contained in the default tag and only contains the minimal packages needed to run Python. If you are working in an environment where only the python image will be deployed and you have space constraints, we recommend to use this one.

See [official python Docker image](https://hub.docker.com/_/python/) for more details.

## Usage

See [detailed usage documentation](http://doc.cytomine.be/display/ALGODOC/Data+access+using+Python+client#DataaccessusingPythonclient-Usage).

### Basic example
Three parameters are required to connect:
* `HOST`: The full URL of Cytomine core (e.g. “http://demo.cytomine.be”).
* `PUBLIC_KEY`: Your cytomine public key.
* `PRIVATE_KEY`: Your cytomine private key. 

First, the connection object has to be initialized.   
    
    from cytomine import Cytomine
    host = "demo.cytomine.be"
    public_key = "XXX" # check your own keys from your account page in the web interface
    private_key = "XXX"
    
    cytomine = Cytomine.connect(host, public_key, private_key)
    
  

The next sample code should print “Hello {username}” where {username} is replaced by your Cytomine username and print the list of available projects.

    from cytomine.models import ProjectCollection
    print("Hello {}".format(cytomine.current_user))
    projects = ProjectCollection().fetch()
    print(projects)
    for project in projects:
        print(project)
        
### Other examples
* [Scripts in examples directory](https://github.com/Cytomine-ULiege/Cytomine-python-client/tree/master/examples)
* [Documentation by examples](http://doc.cytomine.be/display/ALGODOC/Data+access+using+Python+client#DataaccessusingPythonclient-Usage)

## References
When using our software, we kindly ask you to cite our website url and related publications in all your work (publications, studies, oral presentations,...). In particular, we recommend to cite (Marée et al., Bioinformatics 2016) paper, and to use our logo when appropriate. See our license files for additional details.

- URL: http://www.cytomine.org/
- Logo: [Available here](https://cytomine.coop/sites/cytomine.coop/files/inline-images/logo-300-org.png)
- Scientific paper: Raphaël Marée, Loïc Rollus, Benjamin Stévens, Renaud Hoyoux, Gilles Louppe, Rémy Vandaele, Jean-Michel Begon, Philipp Kainz, Pierre Geurts and Louis Wehenkel. Collaborative analysis of multi-gigapixel imaging data using Cytomine, Bioinformatics, DOI: [10.1093/bioinformatics/btw013](http://dx.doi.org/10.1093/bioinformatics/btw013), 2016. 

## License

Apache 2.0