from django.shortcuts import render

from django.http.response import JsonResponse

from .logic import gaussian_blur as gb

import sys

# Create your views here.

def home(request):
    return render(request, 'core/index.html')


def apply_gauss(request):
    
    try:
        imgPath = request.GET.get('img')
        kernel = int(request.GET.get('kernel'))
        sigma = float(request.GET.get('sigma'))

        data = gb.run_with_cuda(imgPath, kernel, sigma)
        data['response'] = 'success'
        print('[INFO]', data)
        return JsonResponse(data)

    except:
        print('[ERROR] apply_gauss')
        print(sys.exc_info())
        return JsonResponse({'response': 'error', 'reason': str(sys.exc_info())})