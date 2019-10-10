#!/usr/bin/env python3

import sys
import csv
import urllib

with open(sys.argv[1]) as f:
    c = csv.reader(f, delimiter=",")
    for line in c:
        name = line[0]
        url = line[1]
        if not (
            url.startswith("https://i.redd.it") or url.startswith("https://imgur.com")
        ):
            continue
        if url[-4:] in [".jpg", ".png"]:
            name = name + url[-4:]
        else:
            continue

        command = ["wget", url, "-O " + '"' + name + '"']
        print(command)
