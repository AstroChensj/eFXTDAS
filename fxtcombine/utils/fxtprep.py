#!/usr/bin/env python3
"""
Copied from fxtchain.
"""
import os
import re
from enum import Enum


FILE_PATTERS = [
    r'''
        (?P<fileNameWithoutVersion>fxt\_(?P<module>\S)\_(?P<obsID>\d{11})
        \_(?P<mode>(ff|pw|tm|df|dt|dp|sv|ds))\_(?P<filter>(00|01|02|03|04|05|80|81|82|83|84|85))\_(?P<pp>po)
        \_(?P<level>\S{2})\_(?P<fileType>(evt|fsaevt))\_)(?P<version>[0-9][0-9a-zA-Z][a-z])
        \.(fits|FITS)
    ''',
    r'''
        (?P<fileNameWithoutVersion>fxt\_\d{11}\_(?P<fileType>mkf)\_)
        (?P<version>[0-9][a-z]{2})\.(fits|FITS)
    ''',
    r'''
        (?P<fileNameWithoutVersion>ep\_\d{11}\_(?P<fileType>att)\_)
        (?P<version>[0-9][a-z]{2})\.(fits|FITS)
    
    ''',
    r'''
        (?P<fileNameWithoutVersion>ep\_\d{11}\_(?P<fileType>orb)\_)
        (?P<version>[0-9][a-z]{2})\.(fits|FITS)
    
    '''
]
EVT_PROFILES = ['module', 'obsID', 'mode', 'filter', 'pp', 'level', 'fileType']


class FxtFileEnum(Enum):
    """
    Enumeration of file types and modules for FXT processing.
    """
    FILE_TYPE_EVT = 'evt'
    FILE_TYPE_FSAEVT = 'fsaevt'
    FILE_TYPE_MKF = 'mkf'
    FILE_TYPE_ATT = 'att'
    FILE_TYPE_ORB = 'orb'

    FILE_MODULE_A = 'a'
    FILE_MODULE_B = 'b'

    FILE_MODE_FF = 'ff'
    FILE_MODE_PW = 'pw'
    FILE_MODE_TM = 'tm'
    

def get_matched_group(match, group_id):
    """
    get matched group from regex search result
    :param match: search result
    :param group_id: group
    :return: group value, None if group not exist
    """
    try:
        return match.group(group_id)
    except IndexError:
        return None
    

class FxtEvtVersion:
    """
    Version:3 characters of ZYX to identify data version,
    Z indicate whether the data is one whole obsID data,
    0 represent the data is not completed while 1 represent
    the data is completed. Y indicate processing number
    of this obsID data, range from a-z,A-Z,0-9. X indicate
    processing software version, range from a-z.The function
    is used to check if the newOne is a newer version
    """
    def __init__(self, version):
        self._version = version

    def compare_code(self):
        """
        Calculate and return a code based on the version attribute.
        """
        process_flag = int(round(abs(ord(self._version[1]) - 122) / 17) / 2)
        return self._version[0] + self._version[2] + str(process_flag) + self._version[1]
   
    @staticmethod
    def compare(one, another):
        """
        Compare the  version with the old version and return True if the new version is newer.
        
        Args:
            one (str): The version to be compared.
            another (str): The version to be compared against.
        
        Returns:
            bool: True if the one is newer, False otherwise.
        """
        return FxtEvtVersion(one).compare_code() > FxtEvtVersion(another).compare_code()


def attr_filter(attribute, tag):
    """
    Filter the attribute based on the provided tag.

    Args:
        attribute: The attribute to be filtered.
        tag: The tag used for filtering.

    Returns:
        bool: True if the attribute matches the tag, False otherwise.
    """
    return (tag is None) or ((str(attribute)).lower() in tag)


def get_input_files(indir, mode, module, file_types):
    """
    get input files and return a dict
    :param indir: directory for scan
    :param mode: fw
    :param module: a or b
    :return: file dict as follows
    {
        'evt':
        {
            'fxt_a_00000000001_pw_00_po_uf_evt_':
            {
                'filePath':'/fxt/event/fxt_a_00000000001_pw_00_po_uf_evt_1aa.fits',
                'module':'a',
                'obsID':'00000000001',
                'mode':'pw',
                'filter':'00',
                'pp':'po',
                'level':'uf',
                'version':'1aa'
            },
            'fxt_b_00000000001_pw_00_po_uf_evt_':
            {
                'filePath':'/fxt/event/fxt_b_00000000001_pw_00_po_uf_evt_1aa.fits',
                'module':'b',
                'obsID':'00000000001',
                'mode':'sw',
                'filter':'00',
                'pp':'po',
                'level':'uf',
                'version':'1aa'
            }
        },
        'mkf':
        {
            'fxt_00000000001_mkf_':
            {
                'filePath':'/fxt/auxil/fxt_00000000001_mkf_1aa.fits',
                'version':'1aa'
            }
        },
        'att':
        {
            'fxt_00000000001_att_':
            {
                'filePath':'/fxt/auxil/fxt_0x0000001001_att_1aa.FITS',
                'version':'1aa'
            }
        },
        'orb':
        {
            'fxt_00000000001_orb_':
            {
                'filePath':'/fxt/auxil/fxt_0x0000001001_orb_1aa.FITS',
                'version':'1aa'
            }
        }
    }
    """
    indir = os.path.abspath(indir)
    # a dict to save return value
    file_dict = {FxtFileEnum.FILE_TYPE_EVT.value: {},
                FxtFileEnum.FILE_TYPE_FSAEVT.value: {},
                FxtFileEnum.FILE_TYPE_MKF.value: {},
                FxtFileEnum.FILE_TYPE_ATT.value: {},
                FxtFileEnum.FILE_TYPE_ORB.value: {}}
    evt_file_path = os.path.join(indir, 'fxt', 'event')
    aux_file_path = os.path.join(indir, 'auxil')
    hk_file_path = os.path.join(indir,'fxt', 'hk')
    evt_file_item = (
        evt_file_path,
        [file for file in os.listdir(evt_file_path) if
         os.path.isfile(os.path.join(evt_file_path, file))] if os.path.exists(evt_file_path) else [])
    aux_file_item = (
        aux_file_path,
        [file for file in os.listdir(aux_file_path) if
         os.path.isfile(os.path.join(aux_file_path, file))] if os.path.exists(aux_file_path) else [])
    hk_file_item = (
        hk_file_path,
        [file for file in os.listdir(hk_file_path) if
         os.path.isfile(os.path.join(hk_file_path, file))] if os.path.exists(hk_file_path) else [])
    items = [evt_file_item, aux_file_item, hk_file_item]
    for root, files in items:
        for file in files:
            for pattern in FILE_PATTERS:
                pattern = re.compile(pattern,  re.VERBOSE)
                match = pattern.search(file)
                if match:
                    file_type = get_matched_group(match, 'fileType')
                    version = get_matched_group(match, 'version')
                    file_name_without_version = get_matched_group(match, 'fileNameWithoutVersion')
                    file_collection = file_dict[file_type]
                    if file_name_without_version in file_collection:
                        # registered
                        pre_lasted_file = file_collection[file_name_without_version]
                        pre_lasted_version = pre_lasted_file['version']
                        if FxtEvtVersion.compare(version, pre_lasted_version):
                            dict_4_upd = {'filePath': os.path.join(root, file), 'version': version}
                            if file_type in [FxtFileEnum.FILE_TYPE_EVT.value, FxtFileEnum.FILE_TYPE_FSAEVT.value]:
                                group_values = [get_matched_group(match, groupId) for groupId in EVT_PROFILES]
                                dict_4_upd.update(dict(zip(EVT_PROFILES, group_values)))
                            pre_lasted_file.update(dict_4_upd)
                    else:
                        # first registration
                        file_dict[file_type][file_name_without_version] = {}
                        file_dict[file_type][file_name_without_version]['filePath'] = os.path.join(root, file)
                        file_dict[file_type][file_name_without_version]['version'] = version
                        if file_type in [FxtFileEnum.FILE_TYPE_EVT.value, FxtFileEnum.FILE_TYPE_FSAEVT.value]:
                            group_values = [get_matched_group(match, groupId) for groupId in EVT_PROFILES]
                            file_properties = dict(zip(EVT_PROFILES, group_values))
                            if attr_filter(file_properties[EVT_PROFILES[0]], module) \
                                    and attr_filter(file_properties[EVT_PROFILES[2]], mode) \
                                    and attr_filter(file_properties[EVT_PROFILES[6]], file_types):
                                # match mode and module
                                file_dict[file_type][file_name_without_version].update(file_properties)
                            else:
                                del file_dict[file_type][file_name_without_version]

    # handle evt files of diagnostic mode(ignore if there are no such files)
    evt_file_dicts = [file_dict[FxtFileEnum.FILE_TYPE_EVT.value], file_dict[FxtFileEnum.FILE_TYPE_FSAEVT.value]]
    for evt_file_dict in evt_file_dicts:
        del_keys = []
        for tmp_evt_key, tmp_evt_dict in evt_file_dict.items():
            tmp_evt_mode = tmp_evt_dict[EVT_PROFILES[2]]
            tmp_evt_filter = tmp_evt_dict[EVT_PROFILES[3]]
            if (tmp_evt_mode in ['df' , 'dt', 'dp', 'sv', 'ds'] and int(tmp_evt_filter) < 10) \
                    or (tmp_evt_mode in ['ff', 'tm', 'pw'] and tmp_evt_filter in ['04', '05']):
                # these are the source diagnostic files, there must be corresponding 
                # files in file_dict ,so delete orginal ones here
                del_keys.append(tmp_evt_key)
        # delete source diagnostic files in evt file dict
        for d_key in del_keys:
            del evt_file_dict[d_key]
    # return the lasted files
    return file_dict
