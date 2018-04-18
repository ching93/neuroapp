from django.db import models


class Subscriber(models.Model):
    email = models.EmailField(primary_key=True)
    name = models.CharField(max_length=128)
    class Meta:
        verbose_name = "Profile"
        verbose_name_plural = "Profiles"