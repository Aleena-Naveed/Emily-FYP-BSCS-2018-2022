import React from "react";
import "./App.css";
import Montserrat from './fonts/Montserrat.ttf';


// import Reptemp from './pages/ReportTemplate';
// import Rep from './components/ReportTemplate'

import { ThemeProvider, createTheme } from '@material-ui/core/styles';
// import Login from "./pages/login";
// import Homepage from "./pages/Homepage";
// import Settings from './pages/Settings'
import Main from './routes/main'
// import PHQ from './pages/ReportTemplate'
// import Dashboard from './pages/dashboard'
const theme = createTheme({
  typography: {
    fontFamily: "Montserrat",
  },
  listItemText: {
    fontFamily: "Montserrat",
  }
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <div>
        <Main />
      </div>
    </ThemeProvider>
  );
}

export default App;
