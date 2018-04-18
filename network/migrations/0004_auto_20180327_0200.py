# Generated by Django 2.0 on 2018-03-26 19:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0003_auto_20180324_1008'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='networkmodel',
            options={'verbose_name': 'Модель НС', 'verbose_name_plural': 'Модели НС'},
        ),
        migrations.RemoveField(
            model_name='weight',
            name='id',
        ),
        migrations.AddField(
            model_name='weight',
            name='name',
            field=models.CharField(default='01', max_length=20, primary_key=True, serialize=False),
        ),
    ]
