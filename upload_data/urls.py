from django.urls import path


from upload_data import views

urlpatterns = [
    path('', views.index, name='index'),
]