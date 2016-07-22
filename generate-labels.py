import sys
import csv

def read_csv_fields(filename, substrings):
    substrings = [subs.lower() for subs in substrings]
    f = open(filename,'rU')
    reader = csv.reader(f, delimiter=',', quotechar='"')
    header = reader.next()
    header = map(lambda x: x.strip().lower(), header)
    hit = [any([(field.find(subs)!=-1) for subs in substrings]) for field in header]
    hidx = filter(lambda i: hit[i], range(len(header)))
    header = map(lambda i: header[i], hidx)
    data = [map(lambda i: row[i], hidx) for row in reader]
    f.close()
    hdata = list()
    hdata.append(header)
    hdata.extend(data)
    hdata = [[field.lower() for field in row] for row in hdata]
    return hdata

def tryconvert(field, d):
    if field in d:
        return d[field]
    else:
        return field

def generate_label_dictionary(master_dataset_csv_file, field_substrings):
    field_substrings.insert(0, 'article id')
    hdata = read_csv_fields(master_dataset_csv_file, field_substrings)
    bdict = {'yes': 1, 'no': 0}
    bdata = [[tryconvert(field, bdict) for field in row] for row in hdata]
    datadict = dict()
    for row in bdata[1:]:
        datadict[row[0]] = row[1:]
    return datadict

def filename_to_article_id(filename):
    filename = filename[0:filename.rfind('.')]
    aid = filename[filename.rfind('.')+1:]
    aid = aid.lower().replace('_',' ')
    return aid

# Main program
field_substrings = ['badge for open data', 'data are available']
master_dataset_csv_file = '/Users/Ina/Dropbox/Projects/Redaptor_Psychology/Gold Standard Psychology/Master Dataset Filtered.csv'
datadict = generate_label_dictionary(master_dataset_csv_file, field_substrings)
print datadict[filename_to_article_id('Journal_of_Personality_and_Social_Psychology.2013.October.10-10-2013_JPSP.txt')]