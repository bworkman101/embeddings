import gzip

# archive_file = 'comm_use.A-B'
archive_file = 'comm_use.C-H'
# archive_file = 'comm_use.I-N'
# archive_file = 'comm_use.O-Z'
# archive_file = 'non_comm_use.A-B'
# archive_file = 'non_comm_use.C-H'
# archive_file = 'non_comm_use.I-N'
# archive_file = 'non_comm_use.O-Z'

with gzip.open('/home/bworkman/development/embeddings/articles/{}.text.gz'.format(archive_file), 'rb') as gz_file:
    for line in gz_file:
        print(line)