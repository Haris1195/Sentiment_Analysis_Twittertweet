from django.urls import path
from . import views
urlpatterns = [
    # path('', views.sentiment_frentend, name = 'sentiment_frentend'),
    # path('Sentiment_backend_databse', views.Sentiment_backend_databse, name = 'Sentiment_backend_databse')
    path('sentiment_frentend', views.sentiment_frentend, name = 'sentiment_frentend'),
    path('Sentiment_backend_databse', views.Sentiment_backend_databse, name = 'Sentiment_backend_databse'),
    # path('login/', views.login, name='login'),
    path('', views.home2, name='home2'),
    path('bar', views.bar, name='bar'),
    path('comparison', views.comparison, name='comparison'),
    path('bar_graph', views.bar_graph, name='bar_graph')
    # path('contact', views.contact , name ='contact'),
    # path('final_output', views.final_output, name = "final_output")

]