
$(function() {
    $("#model-select-btn").on('click',function(e){
        modelName = $("#init-form .model-select select")[0].value;
        url=$("#init-form").attr('action');
        data1={};
        data1['csrfmiddlewaretoken']=$('#init-form [name="csrfmiddlewaretoken"]').val();
        data1['model']=modelName;
        data1['action']='model-select';
        to_send={url: url,
            method: 'POST',
            data: data1,
            success: function(data) {
                //alert('Oh my god SUCCESS!');
                weight_div = $("#init-form .weight-select")[0];
                if ($(weight_div).find("select").length==0) {
                    select = document.createElement('select');
                    weight_div.appendChild(select);
                    weight_div.removeAttribute('hidden');
                }
                else {
                    select = $(weight_div).find("select")[0];
                    select.innerHTML="";
                }
                option = document.createElement('option'); option.innerHTML='Создать новый набор весов';
                option.setAttribute("value","0");
                select.appendChild(option);
                for (key of Object.keys(data)) {
                    option = document.createElement('option');
                    option.value=key; option.innerHTML=key;
                    select.appendChild(option);
                }
            },
            error: function(data) {
                alert(JSON.stringify(data));
            }
        };
        $.ajax(to_send);
        return false;
    });
    $("#start-learn-btn").on('click',function(e){
        url = $("#init-form").attr("action");
        data = get_values("init-form");
        batch_num = Math.ceil((data['till']-data['from'])/data['batch-size']);
        $("#status .epochs").html(data['epochs']);
        $("#status .batches").html(batch_num);

        data['csrfmiddlewaretoken']=$('#init-form [name="csrfmiddlewaretoken"]').val();
        data['action']='train';
        $("#status").removeAttr("hidden");
        $("#status .message").html("Подключение к сокету...");
        var chatSocket = new WebSocket('ws://'+window.location.host+url);

        chatSocket.onmessage =  function(e){
            console.log(e.data);
            data = JSON.parse(e.data)
            switch (data['action']) {
            case 'train-begin':
                $("#status .error-val").html("0");
                $("#status .acc-val").html("0");
                $("#status .message").html("Обучение началось.");
                break;
            case 'train-end':
                $("#status .message").html("Обучение закончено. Проверка точности на тестовом множестве.");
                $("#plot-div").html();
                //$("#plot-div img").attr("src","static/images/train_pic.jpg");
                break;
            case 'test-end':
                $("#status .test-result").removeAttr("hidden");
                $("#status .message").html("Тестирование завершено.");
                $("#status .test-err").html(data['av-err']);
                $("#status .percent").html(data['percent']);
                break;
            case 'epoch-begin':
                $("#status .current-epoch").html(data['epoch']);
                break;
            case 'batch-end':
                $("#status .current-batch").html(data['batch']);
                $("#status .error-val").html(data['loss']);
                $("#status .acc-val").html(data['acc']);
                $("#plot-div img").attr("src","static/images/train_pic.jpg?"+Math.random());
                break;
            default:
                $("#status").attr("hidden","true");
                show_message(data["message"],0);
            } 
        }
        chatSocket.onclose = function(data) {
            console.log("socket closed: %s" % data);
        }
        chatSocket.onopen = function(e) {
            chatSocket.send(JSON.stringify(data));
        }

    });
});



function get_values(id) {
    res={}
    for (item of $("#"+id+" [name]"))
        res[item.name]=item.value;
    return res;
}

function subscribe(url) {
  var xhr = new XMLHttpRequest();

  xhr.onreadystatechange = function() {
    if (this.readyState != 4) {
        console.log("training ended");
        return;
    }

    if (this.status == 200) {
        console.log(xhr.responseText);
    } else {
        throw 'Error';
    }

    data['action']='state-check';
    sleep(2000)
    subscribe(url,data);
  }
  xhr.open("POST", url, true);
  xhr.send(data);
}