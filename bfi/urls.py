"""
URL configuration for bfi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path,include
from myapp import views
from django.contrib import admin
from django.urls import path, include
from django.views.i18n import set_language
from django.urls import path
from django.views.generic import TemplateView



urlpatterns = [
    path('admin/', admin.site.urls),
    path('i18n/', set_language, name='set_language'),
    path('', TemplateView.as_view(template_name='base.html'), name='home'),
    path('i18n/', include('django.conf.urls.i18n')),
    path('', include('myapp.urls', namespace='myapp')),


   
]

