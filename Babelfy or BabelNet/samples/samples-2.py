import urllib2
import urllib
import json
import gzip

from StringIO import StringIO

service_url = 'https://babelfy.io/v1/disambiguate'

text = 'BabelNet is both a multilingual encyclopedic dictionary and a semantic network'
lang = 'EN'
key = '5e962130-b37f-4105-8512-4c97b4f3cb30'

params = {
    'text': text,
    'lang': lang,
    'key': key
}

url = service_url + '?' + urllib.urlencode(params)
request = urllib2.Request(url)
request.add_header('Accept-encoding', 'gzip')
response = urllib2.urlopen(request)

if response.info().get('Content-Encoding') == 'gzip':
    buf = StringIO(response.read())
    f = gzip.GzipFile(fileobj=buf)
    data = json.loads(f.read())

    # retrieving data
    for result in data:
        # retrieving token fragment
        tokenFragment = result.get('tokenFragment')
        tfStart = tokenFragment.get('start')
        tfEnd = tokenFragment.get('end')
        print str(tfStart) + "\t" + str(tfEnd)

        # retrieving char fragment
        charFragment = result.get('charFragment')
        cfStart = charFragment.get('start')
        cfEnd = charFragment.get('end')
        print str(cfStart) + "\t" + str(cfEnd)

        # retrieving BabelSynset ID
        synsetId = result.get('babelSynsetID')
        print synsetId
