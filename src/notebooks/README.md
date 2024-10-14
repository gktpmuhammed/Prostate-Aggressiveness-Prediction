# RUN JUPYTE NOTEBOOKS

In the ssh terminal:

```
jupyter notebook --no-browser --port=<available_port>
```

This will return a http://localhost... URL

In the local terminal:

```
ssh -L <available_port>:localhost:<available_port> <user>@131.159.110.3
```

Click on the URL

