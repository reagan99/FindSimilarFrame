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

