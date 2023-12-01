function scroll_down() {
    const list = document.querySelectorAll('li');
    list.forEach(li => {
        li.style.transform = 'translateY(-100%)';
    });
}

function scroll_up() {
    const list = document.querySelectorAll('li');
    list.forEach(li => {
        li.style.transform = 'translateY(0%)';
    });
    location.reload(true);
}

document.addEventListener('DOMContentLoaded', (event) => {
    // 웹캠 지원 여부 확인
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      // 웹캠에서 비디오 스트림 얻기
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          const webcam = document.getElementById('webcam');
          webcam.srcObject = stream;
        })
        .catch((error) => {
          console.error('웹캠을 사용할 수 없습니다:', error);
        });
    } else {
      console.error('미디어 장치 API가 지원되지 않습니다.');
    }
  });