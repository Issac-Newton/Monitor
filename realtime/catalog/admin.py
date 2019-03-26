from django.contrib import admin
from .models import realtimeNodeInfo,cpuutilInfo
# Register your models here.



admin.site.register(realtimeNodeInfo)
admin.site.register(cpuutilInfo)

admin.site.unregister(realtimeNodeInfo)
admin.site.unregister(cpuutilInfo)
@admin.register(realtimeNodeInfo)
class realtimeNodeInfoAdmin(admin.ModelAdmin):
    """
    Administration object for Author models.
    Defines:
     - fields to be displayed in list view (list_display)
     - orders fields in detail view (fields), grouping the date fields horizontally
     - adds inline addition of books in author view (inlines)
    """
    list_display = ('runUser', 'idleNode', 'pendJob', 'availableCore')

@admin.register(cpuutilInfo)
class cpuutilInfoAdmin(admin.ModelAdmin):
    """
    Administration object for Author models.
    Defines:
     - fields to be displayed in list view (list_display)
     - orders fields in detail view (fields), grouping the date fields horizontally
     - adds inline addition of books in author view (inlines)
    """
    list_display = ('casnw', 'dicp', 'era', 'erai', 'gspcc', 'hku', 'hust', 'iapcm', 'nscccs', 'nsccgz', 'nsccjn', 'jscctj', 'nsccwx', 'siat', 'sjtu', 'ssc', 'ustc', 'xjtu', 'anomaly')