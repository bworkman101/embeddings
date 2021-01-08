
import gensim
import gzip

class ArchiveIter:
    def __init__(self, archive):
        self.archive = archive

    def __iter__(self):
        return self

    def __next__(self):
        line = self.archive.__next__()
        return gensim.utils.simple_preprocess(line)

# archive_files = ['comm_use.A-B']
archive_files = ['biorxiv_medrxiv', 
                 'comm_use_subset', 
                 'noncomm_use_subset', 
                 'pmc_custom_license']

def gzip_iter(archive_file):
    gz_file = gzip.open('/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/texts/{}.text.gz'.format(archive_file), 'rb')
    return ArchiveIter(gz_file)

for archive_file in archive_files:
    filename = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/texts_clean/{}.text.gz'.format(archive_file)
    # filename = '/home/bworkman/development/embeddings/articles/{}.cor.gz'.format(archive_file)
    print('creating corpus {}'.format(archive_file))
    gensim.utils.save_as_line_sentence(gzip_iter(archive_file), filename)