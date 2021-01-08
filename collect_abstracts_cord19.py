import gzip
import tarfile
from contextlib import closing
import re
import json

class ArchiveIter:
    def __init__(self, archive, file_name):
        self.archive = archive
        self.file_name = file_name

    def __iter__(self):
        return self

    def __next__(self): # Python 3: def __next__(self)
        while True:
            body = self._extract_next_body()
            if body is not None:
                if body == "stop":
                    return None
                else:
                    return body

    def _extract_next_body(self):
        print("extracting body of", self.file_name)
        member = self.archive.next()

        def get_text(json_doc_file):
            json_doc = json.loads(json_doc_file)
            title = json_doc['metadata']['title']

            def collect_texts(text_objs):
                texts = []
                for text_obj in text_objs:
                    texts.append(text_obj['text'])
                return ' '.join(texts)

            abstract = collect_texts(json_doc['abstract'])
            body = collect_texts(json_doc['body_text'])

            return ' '.join([title, abstract, body])

        if member is None:
            return "stop"

        try:
            if member.isreg() and member.name.endswith('.json'):
                with closing(archive.extractfile(member)) as json_file_obj:
                    print("extracting {}".format(member))
                    text = get_text(json_file_obj.read().decode("utf-8"))
                    return text
        except Exception as e:
                print("parsing error {}".format(e))
                raise e

        return None


archive_files = ['biorxiv_medrxiv', 
                 'comm_use_subset', 
                 'noncomm_use_subset', 
                 'pmc_custom_license']

for archive_file in archive_files:
    print("parsing archive {}".format(archive_file))
    read_archive = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/{}.tar.gz'.format(archive_file)
    write_archive = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/texts/{}.text.gz'.format(archive_file)
    print("reading", read_archive)
    print("writing", write_archive)
    with tarfile.open(read_archive) as archive:
        with gzip.open(write_archive, 'wb') as out:
            archive_iter = ArchiveIter(archive, archive_file)
            for body in archive_iter:
                if body is None:
                    break
                out.write(body.encode('utf-8') + '\n'.encode('utf-8'))