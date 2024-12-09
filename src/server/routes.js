const { postPredictHandler, getPredictHistoriesHandler } = require('../server/handler');
 
const routes = [
  {
    path: '/predict',
    method: 'POST',
    handler: postPredictHandler,
    options: {
      payload: {
        allow: 'multipart/form-data',
        multipart: true,
        maxBytes: 2000000
      }
    }
  }
]
 
module.exports = routes;