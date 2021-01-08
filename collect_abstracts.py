import gzip
import tarfile
from contextlib import closing
from xml.etree import ElementTree as etree
import re

class ArchiveIter:
    def __init__(self, archive, file_name):
        self.archive = archive
        self.file_name = file_name

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        while True:
            body = self._extract_next_body()
            if body is not None:
                if body == "stop":
                    return None
                else:
                    return body

    def _extract_next_body(self):
        member = self.archive.next()

        def get_text(xml_node):
            if xml_node is not None:
                text = ' '.join(xml_node.itertext())
                return text
            return None

        if member is None:
            return "stop"

        try:
            if member.isreg() and member.name.endswith('.nxml'):
                with closing(archive.extractfile(member)) as xmlfile:
                    root = etree.parse(xmlfile).getroot()
                    title = '.'.join(e.text for e in root.findall('front/article-meta/article-id'))
                    abstract = root.find('abstract')
                    body = root.find('body')
                    abstract_text = get_text(abstract)
                    body_text = get_text(body)
                    print("{} extracting {}".format(self.file_name.encode('utf-8'), title.encode('utf-8')))

                    if abstract_text is not None and body_text is not None:
                        return abstract_text + " " + body_text
                    
                    if abstract_text is None and body_text is not None:
                        return body_text

                    if abstract_text is not None and body_text is None:
                        return abstract_text
        except:
                print("parsing error")

        return None


archive_files = [#'comm_use.A-B',
                #  'comm_use.C-H',
                #  'comm_use.I-N',
                #  'comm_use.O-Z',
                #  'non_comm_use.A-B',
                #  'non_comm_use.C-H',
                #  'non_comm_use.I-N',
                 'non_comm_use.O-Z',
                 ]

for archive_file in archive_files:
    print("parsing archive {}".format(archive_file))
    with tarfile.open('/home/bworkman/Documents/pubmed/{}.xml.tar.gz'.format(archive_file)) as archive:
        with gzip.open('/home/bworkman/development/embeddings/articles/{}.text.gz'.format(archive_file), 'wb') as out:
            archive_iter = ArchiveIter(archive, archive_file)
            for body in archive_iter:
                if body is None:
                    break
                out.write(body.encode('utf-8').replace('\n', ' ') + '\n'.encode('utf-8'))