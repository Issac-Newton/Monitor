from django.shortcuts import render

# Create your views here.

from .models import cpuutilInfo
from .getdata import get_data
from .util import *
import numpy as np

predict_data = np.array(18, np.int32)
def showdata(request):
    """
    View function for home page of site.
    """
    # Generate counts of some of the main objects
    now_data = np.zeros(18)
    jsdata = get_data(url = 'http://api.cngrid.org/v2/show/cngrid/realtimeInfo')
    dict,status = from_json_to_dict(jsdata)
    if status != 2:
        global predict_data
        now_data, predict_data,status = getstatus(dict,predict_data)
        a_record = cpuutilInfo(casnw = dict['casnw'],dicp = dict['dicp'],era = dict['era'],erai = dict['erai'] , hku = dict['hku'], \
                                hust = dict['hust'], iapcm = dict['iapcm'], nscccs = dict['nscccs'], nsccgz = dict['nsccgz'], \
                                nsccjn = dict['nsccjn'], nscctj = dict['nscctj'], nsccwx = dict['nsccwx'], siat = dict['siat'], \
                                sjtu = dict['sjtu'], ssc = dict['ssc'], ustc = dict['ustc'], xjtu = dict['xjtu'], anomaly = status)
        # Save the object into the database.
        a_record.save()
    else:
        a_record = cpuutilInfo(casnw= 0, dicp = 0, era= 0, erai=0, hku=0, \
                               hust=0, iapcm=0, nscccs=0, nsccgz=0, \
                               nsccjn=0, nscctj=0, nsccwx=0, siat=0, \
                               sjtu=0, ssc=0, ustc=0, xjtu = 0, anomaly=status)
        # Save the object into the database.
        a_record.save()
    # Render the HTML template index.html with the data in the context variable.
    return render(
        request,
        'index.html',
        context={'now_data': now_data, 'predict_data': predict_data},
    )