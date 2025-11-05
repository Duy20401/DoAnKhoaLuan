from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('alphabet/', views.learn_alphabet, name='learn_alphabet'),
    path('alphabet/<str:letter>/', views.alphabet_detail, name='alphabet_detail'),
    path('words/', views.learn_words, name='learn_words'),
    path('words/<str:word>/', views.word_detail, name='word_detail'),
    path('practice/', views.practice, name='practice'),
    path('practice/camera/', views.practice_camera, name='practice_camera'),
    
    # API for real-time recognition
    path('api/recognize/', views.api_recognize, name='api_recognize'),
    path('practice/words/camera/', views.practice_words_camera, name='practice_words_camera'),
    path('api/recognize/words/', views.api_recognize_words, name='api_recognize_words'),
]