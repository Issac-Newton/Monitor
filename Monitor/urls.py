"""Monitor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from testStatic import views as test_views
from begin import views as begin_views
from logMon import views as logMon_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('test/',test_views.index),
    path('begin/',begin_views.index),
    path('logMon/',logMon_views.index),
    path('get_log_data/',logMon_views.get_log_data),
    path('mosaic_chart/',logMon_views.mosaic_chart),
    path('cluster_info/',logMon_views.cluster_info),
    path('user_info/',logMon_views.user_info),
]
