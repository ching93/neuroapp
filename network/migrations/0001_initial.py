# Generated by Django 2.0 on 2018-02-06 12:11

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='LayerInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('kernelAmount', models.IntegerField(default=1)),
                ('kernelSizeX', models.IntegerField(default=1)),
                ('kernelSizeY', models.IntegerField(default=1)),
            ],
            options={
                'verbose_name': 'Информация о слое',
                'verbose_name_plural': 'Информация о слоях',
            },
        ),
        migrations.CreateModel(
            name='LayerType',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(default='full-connected', max_length=20)),
            ],
            options={
                'verbose_name': 'Тип слоя',
                'verbose_name_plural': 'Типы слоев',
            },
        ),
        migrations.CreateModel(
            name='NnModel',
            fields=[
                ('name', models.CharField(max_length=20, primary_key=True, serialize=False)),
            ],
            options={
                'verbose_name': 'Модель НН',
                'verbose_name_plural': 'Модели НН',
            },
        ),
        migrations.AddField(
            model_name='layerinfo',
            name='layerType',
            field=models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='network.LayerType'),
        ),
        migrations.AddField(
            model_name='layerinfo',
            name='model',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='network.NnModel'),
        ),
    ]
