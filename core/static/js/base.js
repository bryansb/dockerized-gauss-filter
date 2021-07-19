function process(){
    document.getElementById('process-btn').disabled = true;

    let imgPath = document.getElementById("img-select").value;
    let kernel = document.getElementById("kernel-select").value;
    let sigma = document.getElementById("sigma-value").value;

    if (imgPath !== "null" && kernel !== "null") {
        let url = 'http://127.0.0.1:8888/process';
        $.get(url, { img: imgPath, kernel: kernel, sigma: sigma}).done( function( data ) {
            
            print(data)
            if (data['response'] === 'success') {
                document.getElementById("img-result-process").src = data['imgUrl'] + '?' + new Date().getTime();
            } else {
                document.getElementById("img-result-process").src = "/static/img/empty.png";
            }
            document.getElementById('process-btn').disabled = false;
        });
    } else {
        document.getElementById('process-btn').disabled = false;
    }
}


function print(data) {
    var html = '';
    if (data['response'] === 'success') {
        html = `
        <tbody>
            <tr>
                <th scope="row" class="text-right">Tiempo</th>
                <td>${data['final_time']}</td>
            </tr>
            <tr>
                <th scope="row" class="text-right">Máscara</th>
                <td>${data['kernel']}</td>
            </tr>
            <tr>
                <th scope="row" class="text-right">Desviación Estándar</th>
                <td>${data['sigma']}</td>
            </tr>
            <tr>
                <th scope="row" class="text-right">Número de Bloques</th>
                <td>${data['bdim']}</td>
            </tr>
            <tr>
                <th scope="row" class="text-right">Cuadrícula</th>
                <td>${data['gdim']}</td>
            </tr>
        </tbody>
        `;
    } else {
        html = `
        <tbody>
            <tr>
                <th scope="row" class="text-right">Error</th>
                <td>${data['reason']}</td>
            </tr>
        </tbody>`;
    }

    document.getElementById("table-result-detail").innerHTML = html;
}

$(function () {
    $('[data-toggle="tooltip"]').tooltip()
})