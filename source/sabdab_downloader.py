from __future__ import with_statement
import shutil
import time
description='''


SAbDab Download Script                         \\\    //
The OPIG Antibody Database                      \\\  //
Authors: James Dunbar and Konrad Krawczyk 2013.   ||
Contributors: Jinwoo Leem                         ||
Supervisor: Charlotte Deane

Contact: James Dunbar (james.dunbar@dtc.ox.ac.uk)
      
In collaboration with:
UCB: Jiye Shi, Terry Baker. 
Roche: Angelika Fuchs, Guy Georges. 


o This is a script that allows a user to download data from SAbDab.
o It requires a csv summary file downloaded from the website (opig.stats.ox.ac.uk/webapps/sabdab)

o This file should contain AT LEAST:

    1. A header line with tab-separated fields as "pdb    Hchain    Lchain    model"
    2. The pdb identifier, heavy chain, light chain and model id on new lines e.g.
    
    pdb    Hchain    Lchain    model
    12e8   H         L         0
    12e8   P         M         0
    1ahw   B         A         0
    1ahw   E         D         0
    .      .         .         .
    .      .         .         .
    .      .         .         .

o Other fields will be ignored but may be included in the file.

o The user must provide a directory in which the data should be downloaded to. 
o The type of data that should be downloaded should be specified using the command-line options.

o Example useage:
    To run on a linux command line type:

    python sabdab_downloader -s summary_file.csv -o path/to/output/ --original_pdb 
    
    This will create a directory in  "path/to/output/" name sabdab_dataset. 
    It will contain a directory for each unique pdb code in the summary_file.csv .
    The structure for each of these pdbs will be downloaded there. 
 
'''

epilogue="""
Copyright (C) 2013 James Dunbar
"""


import argparse, sys, os, urllib

TRIES = 3
REST = 0.2

def getpdb(pdb_entry, out_path):
    """
    Get the PDB file from sabdab.     
    Check that it has successfully downloaded.
    """
    out_file = os.path.join( out_path, "%s.pdb"%pdb_entry)
    if os.path.exists(out_file):
        return True
    urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/structure/%s.pdb"%(pdb_entry,pdb_entry), out_file)
    if os.path.isfile(out_file):
        Retrieved = open(out_file).read()
        if not Retrieved.count("ATOM"):
            print "Failed to retrieve PDB file from SAbDab"
            os.remove(out_file)
            raise Exception
        else:
            return True
    else:
        raise Exception

def getpdb_retry(pdb_entry, out_path):
    got_data = False
    for i in range(TRIES):
        print 'pdb try ' + str(i)
        try:
            got_data = getpdb(pdb_entry, out_path)
        except Exception:
            time.sleep(REST)
            continue
        break
    return got_data

def getchothpdb(pdb_entry, out_path):
    """
    Get the chothia PDB file from sabdab.     
    Check that it has successfully downloaded.
    """
    out_file = os.path.join( out_path, "%s.pdb"%pdb_entry)
    if os.path.exists(out_file):
        return True
    urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/structure/chothia/%s.pdb"%(pdb_entry,pdb_entry), out_file)
    if os.path.isfile(out_file):
        Retrieved = open(out_file).read()
        if not Retrieved.count("ATOM"):
            print "Failed to retrieve chothia PDB file from SAbDab"
            os.remove(out_file)
            raise Exception
        else:
            return True
    else:
        raise Exception

def getchothpdb_retry(pdb_entry, out_path):
    got_data = False
    for i in range(TRIES):
        print 'chothpdb try ' + str(i)
        try:
            got_data = getchothpdb(pdb_entry, out_path)
        except:
            time.sleep(REST)
            continue
        break
    return got_data

def getsequence(entry, fab_list, out_path):
    """
    Get the sequence files     
    Check that they successfully download
    Put them into the directory
    """
    
    out_file = os.path.join( out_path, "%s_raw.pdb"%entry)
    if not os.path.exists(out_file):
        urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/sequences/%s_raw.fa"%(entry,entry), out_file)
        if os.path.isfile(out_file):
            Retrieved = open(out_file).read()
            if not Retrieved.count(">%s"%entry):
                print "Failed to retrieve sequence file from SAbDab."
                os.remove(out_file)
                raise
        else:
            raise

    for fab in fab_list:
        Hchain = fab[1]
        if Hchain!="NA":
            Hchain = Hchain.upper()
            out_file = os.path.join( out_path, "%s_%s_VH.fa"%(entry,Hchain) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/sequences/%s_%s_VH.fa"%(entry,entry,Hchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count(">%s"%entry):
                        print "Failed to retrieve H chain sequence file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise
        Lchain = fab[2]
        if Lchain!="NA":
            Lchain = Lchain.upper()
            out_file = os.path.join( out_path, "%s_%s_VL.fa"%(entry,Lchain) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/sequences/%s_%s_VL.fa"%(entry,entry,Lchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count(">%s"%entry):
                        print "Failed to retrieve L chain sequence file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise
    return True


def getsequence_retry(entry, fab_list, out_path):
    got_data = False
    for i in range(TRIES):
        print 'sequence try ' + str(i)
        try:
            got_data = getsequence(entry, fab_list, out_path)
        except:
            time.sleep(REST)
            continue
        break
    return got_data

def getannotation(entry, fab_list, out_path):
    """
    Get the annotation files for the antibody sequences.
    These are for the variable region of the sequences only.
    """
    for fab in fab_list:
        Hchain = fab[1]
        if Hchain!="NA":
            Hchain = Hchain.upper()
            out_file = os.path.join( out_path, "%s_%s_VH.ann"%(entry,Hchain) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/annotation/%s_%s_VH.ann"%(entry,entry,Hchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count("H3"):
                        print "Failed to retrieve Hchain annotation file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise

        Lchain = fab[2]
        if Lchain!="NA":
            Lchain = Lchain.upper()
            out_file = os.path.join( out_path, "%s_%s_VL.ann"%(entry,Lchain) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/annotation/%s_%s_VL.ann"%(entry,entry,Lchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count("L3"):
                        print "Failed to retrieve Lchain annotation file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise
    return True

def getannotation_retry(entry, fab_list, out_path):
    got_data = False
    for i in range(TRIES):
        print 'annotation try ' + str(i)
        try:
            got_data = getannotation(entry, fab_list, out_path)
        except:
            time.sleep(REST)
            continue
        break
    return got_data


def getabangle(entry, fab_list, out_path):
    """
    Get the orientation angles for any of the fabs in the pdb.
    A non-paired antibody chain e.g VHH will have NA as the other chain identifier. 
    """
    for fab in fab_list:
        if "NA" in fab:
            continue
        else:
            out_file = os.path.join( out_path, "%s.abangle"%(entry) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/abangle/%s.abangle"%(entry,entry), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count(entry):
                        print "Failed to retrieve abangle file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise
                return True
    return True
  
def getabangle_retry(entry, fab_list, out_path):
    got_data = False
    for i in range(TRIES):
        print 'abangle try ' + str(i)
        try:
            got_data = getabangle(entry, fab_list, out_path)
        except:
            time.sleep(REST)
            continue
        break
    return got_data

def getimgt(entry, fab_list, out_path):
    """
    Get the imgt files for the antibody sequences.
    
    """
    for fab in fab_list:
        Hchain = fab[1]
        if Hchain!="NA":
            Hchain = Hchain.upper()
            out_file = os.path.join( out_path, "%s_%s_H.ann"%(entry,Hchain) )
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/imgt/%s_%s_H.imgt"%(entry,entry,Hchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count("gene_type"):
                        print "Failed to retrieve HChain imgt file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise

        Lchain = fab[2]
        if Lchain!="NA":
            Lchain = Lchain.upper()
            out_file = os.path.join( out_path, "%s_%s_L.ann"%(entry,Lchain))
            if not os.path.exists(out_file):
                urllib.urlretrieve("http://opig.stats.ox.ac.uk/webapps/abdb/web_front/data/entries/%s/imgt/%s_%s_L.imgt"%(entry,entry,Lchain), out_file)
                if os.path.isfile(out_file):
                    Retrieved = open(out_file).read()
                    if not Retrieved.count("gene_type"):
                        print "Failed to retrieve LChain imgt file from SAbDab."
                        os.remove(out_file)
                        raise
                else:
                    raise
    return True

def getimgt_retry(entry, fab_list, out_path):
    got_data = False
    for i in range(TRIES):
        print 'imgt try ' + str(i)
        try:
            got_data = getimgt(entry, fab_list, out_path)
        except:
            time.sleep(REST)
            continue
        break
    return got_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="sabdab_downloader", description=description, epilog=epilogue,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument( '--summary_file','-s',type=str, help="A tab-separated csv downloaded from SAbDab - opig.stats.ox.ac.uk/webapps/sabdab.", dest="summary_file")
    parser.add_argument( '--output_path','-o',type=str, help="The path to the output directory.", dest="output_path")
    parser.add_argument( '--original_pdb',action="store_true", help="Download the pdb structure(s).", dest="original_pdb")
    parser.add_argument( '--chothia_pdb', action="store_true", help="Download the chothia re-numbered pdb structure(s).", dest="chothia_pdb")
    parser.add_argument( '--sequences',action="store_true", help="Download the sequence information.", dest="sequence")
    parser.add_argument( '--annotation',action="store_true", help="Download the chothia numbered sequence information.", dest="annotation")
    parser.add_argument( '--abangle',action="store_true", help="Download the abangle angles.", dest="abangle")
    parser.add_argument( '--imgt',action="store_true", help="Download the imgt information for the structure.", dest="imgt")

    args= parser.parse_args()

    if len(sys.argv)<2:
        parser.print_help()
        sys.exit(0)
        
    #####################
    #  Check the inputs #
    #####################

    if not args.summary_file:
        print >> sys.stderr, "No summary file found."
        sys.exit(1)
    if not args.output_path:
        print >> sys.stderr, "No output path given."
        sys.exit(1)

    if not (args.original_pdb or args.chothia_pdb or args.sequence or args.annotation or args.abangle or args.imgt):
        print >> sys.stderr, "No requested data type given. Please choose at least one."

    if not os.path.exists(args.output_path):
        print >> sys.stderr, "Output path does not exist."
        sys.exit(1)

    if not os.path.isdir(args.output_path):
        print >> sys.stderr, "Output path is not a directory."
        sys.exit(1)

    if not os.access(args.output_path, os.W_OK):
        print >> sys.stderr, "Output path is not writable."
        sys.exit(1)

    # Set up output directory
    output_path = os.path.join(args.output_path,"sabdab_dataset")
    try:
        os.mkdir(output_path)
    except OSError:
        pass

    # Get the summary data    
    try:
        with open(args.summary_file,'r') as input_file:
            lines = input_file.readlines()
            header = lines[0].strip().split("\t")[:4]
            if header != ["pdb", "Hchain", "Lchain", "model"]: 
                raise IndexError
            data={}
            for line in lines[1:]:
                if not line.strip(): continue
                entry = line.strip().split("\t")[:4]
                if len(entry) < 4 and not entry[0].isalnum():
                    raise IndexError
                try:
                    data[entry[0].lower()].append(entry)
                except KeyError:
                    data[entry[0].lower()] = [entry]
    except IOError:
        print >> sys.stderr, "Could not open summary file."
        sys.exit(1)
    except IndexError:
        print >> sys.stderr, "Summary file in incorrect format."
        sys.exit(1)
        
    
    for pdb_entry in data:
        print "Getting data for %s"%pdb_entry
        got_data=True
        pdb_entry_dir = os.path.join(output_path, pdb_entry)
        try:
            os.mkdir(pdb_entry_dir)
        except:
            pass
        if args.original_pdb or args.chothia_pdb:
            struc_out_path = os.path.join(pdb_entry_dir,"structure")
            try:
                os.mkdir(struc_out_path)
            except:
                pass
            if args.original_pdb:
                if not getpdb_retry(pdb_entry, struc_out_path):
                    got_data=False
                
            if args.chothia_pdb:
                choth_struc_out_path = os.path.join(struc_out_path,"chothia")
                try:
                    os.mkdir(choth_struc_out_path)
                except:
                    pass
                if not getchothpdb_retry(pdb_entry, choth_struc_out_path):
                    got_data=False
 
        if args.sequence:
            seq_out_path = os.path.join(pdb_entry_dir,"sequence")
            try:
                os.mkdir(seq_out_path)
            except:
                pass
            if not getsequence_retry(pdb_entry, data[pdb_entry] , seq_out_path):
                got_data=False
                
        if args.annotation:
            annotation_out_path = os.path.join(pdb_entry_dir,"annotation")
            try:
                os.mkdir(annotation_out_path)
            except:
                pass
            if not getannotation_retry(pdb_entry, data[pdb_entry] , annotation_out_path):
                pass
                
        if args.abangle:
            abangle_out_path = os.path.join(pdb_entry_dir,"abangle")
            try:
                os.mkdir(abangle_out_path)
            except:
                pass
            if not getabangle_retry(pdb_entry, data[pdb_entry] , abangle_out_path):
                pass

        if args.imgt:
            imgt_out_path = os.path.join(pdb_entry_dir,"imgt")
            try:
                os.mkdir(imgt_out_path)
            except:
                pass
            if not getimgt_retry(pdb_entry, data[pdb_entry] , imgt_out_path):
                pass

        if not got_data:
            try:
                shutil.rmtree(pdb_entry_dir)
            except:
                print("what?")
                pass

        
        
    
    
                
    
    






