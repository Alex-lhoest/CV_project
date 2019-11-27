# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from argparse import ArgumentParser

import os

from cytomine import Cytomine
from cytomine.models.image import ImageInstanceCollection
from cytomine.models import ProjectCollection

import pandas as pd

__author__ = "Rubens Ulysse <urubens@uliege.be>"

# This example script allows you to get the list of images (metadata) in a given project.
# If a download path is provided, it downloads all original images like they have been uploaded to Cytomine.

if __name__ == '__main__':
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine connection parameters
    parser.add_argument('--cred', dest='cred',
                        default='demo.cytomine.be', help="Your credentials")
    # parser.add_argument('--cytomine_public_key', dest='public_key',
    #                     help="The Cytomine public key")
    # parser.add_argument('--cytomine_private_key', dest='private_key',
    #                     help="The Cytomine private key")

    # Cytomine project ID
    # parser.add_argument('--cytomine_id_project', dest='id_project',
    #                     help="The project from which we want the images")

    # Download path
    parser.add_argument('--download_path', required=False,
                        help="Where to store images")

    params, other = parser.parse_known_args(sys.argv[1:])

    data = pd.read_csv(params.cred, header=None)

    host = data[0][0]
    public_key = data[0][1]
    private_key = data[0][2]

    conn = Cytomine.connect(host, public_key, private_key)

    projects = ProjectCollection().fetch()

    for project in projects:
        with Cytomine(host=host, public_key=public_key, private_key=private_key,
                    verbose=logging.INFO) as cytomine:

            # We want all image instances in a given project.
            # => Fetch the collection of image instances, filtered by the given project.
            image_instances = ImageInstanceCollection().fetch_with_filter("project", project.id)
            print(image_instances)

            for image in image_instances:
                # Every element in the collection is an ImageInstance object.
                # See ImageInstance class for all available properties (width, height, resolution, ...)
                print("Image ID: {} | Width: {} | Height: {} | Resolution: {} | Magnification: {} | Filename: {}".format(
                    image.id, image.width, image.height, image.resolution, image.magnification, image.filename
                ))

                if params.download_path:
                    # To download the original files that have been uploaded to Cytomine
                    # Attributes of ImageInstance are parsed in the filename
                    image.download(os.path.join(params.download_path, str(id), "{originalFilename}"))