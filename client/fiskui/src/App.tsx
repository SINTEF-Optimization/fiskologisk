import { BrowserRouter } from 'react-router-dom';
import { Route, Routes } from 'react-router';
import './App.css';
import { FiskologiskApiServiceProvider } from './services/fiskologiskApi/fiskologiskApiService';
import { HomePage } from './pages/home';

function App() {
  return (
    <BrowserRouter>
      <FiskologiskApiServiceProvider>
        <Routes>
          <Route path="/" element={<HomePage/>}/>
        </Routes>
      </FiskologiskApiServiceProvider>
    </BrowserRouter>
  );
}

export default App;
