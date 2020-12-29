const winston = require('winston');
const express = require('express');
const app = express();

require('./startup/logging');
require('./startup/routes')(app);

const port = process.env.PORT || 8000;
app.listen(port, () => winston.info(`Listening on port ${port}...`));