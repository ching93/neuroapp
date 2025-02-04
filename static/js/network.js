function addLayerType(e) {
    sr = e.target.parentElement;
    node = $(".layerTypeForm")[0].cloneNode(true);

    for (elem of node.children) {
        if (elem.classList.contains('param'))
            elem.setAttribute('hidden','hidden');
        if ($(elem).find('select')) {
            $(elem).find('select').on('change',showFields);
            $(elem).find('select').on('create',showFields);
        }
    }
    sr.parentElement.insertBefore(node,sr);
}

function removeLayer(sr) {
    if ($(".layerTypeForm").length>1)
        sr.parentElement.parentElement.remove();
}

function showFields() {
    src = event.target;
    param1='none';
    param2='none';
    switch (src.value) {
        case 'Convolutional': param1='kernelSize'; param2='kernelAmount'; break;
        case 'Pooling': param1='kernelSize'; break;
        case 'Dense': param1='neuronAmount'; break;
    }
    for (elem of src.parentElement.parentElement.children) {
        if (elem.classList.contains('param'))
            if (elem.classList.contains(param1) || elem.classList.contains(param2))
                elem.removeAttribute('hidden');
            else
                elem.setAttribute('hidden','hidden');
    }
}

$(function(){
    $('.layerType select').on('change',showFields);
    $('.layerType select').on('create',addLayerType);
    $('.btn-add-layer').on('click',addLayerType);
})
