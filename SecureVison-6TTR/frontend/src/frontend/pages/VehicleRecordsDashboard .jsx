import React, { useState, useEffect } from 'react';
import Navbar from '../../frontend/Navbar/Navbar';
import './Dashboard.css';

const VehicleRecordsDashboard = () => {
  const [vehicleRecords, setVehicleRecords] = useState([]);
  const [vehicleLogs, setVehicleLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [logsLoading, setLogsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [recordsPerPage] = useState(10);
  const [activeTab, setActiveTab] = useState('records'); // 'records' or 'logs'
  const [filterDate, setFilterDate] = useState('');
  const [stats, setStats] = useState({
    totalVehicles: 0,
    registeredVehicles: 0,
    detectedToday: 0,
    accessGrantedToday: 0
  });

  useEffect(() => {
    if (activeTab === 'records') {
      fetchVehicleRecords();
    } else {
      fetchVehicleLogs();
    }
    fetchVehicleStats();
  }, [activeTab]);

  const fetchVehicleRecords = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/vehicle_plate/records');
      if (!response.ok) {
        throw new Error('Failed to fetch vehicle records');
      }
      const data = await response.json();
      setVehicleRecords(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchVehicleStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/vehicle_plate/stats');
      if (!response.ok) {
        throw new Error('Failed to fetch vehicle stats');
      }
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching vehicle stats:', err);
    }
  };

  const fetchVehicleLogs = async () => {
    try {
      setLogsLoading(true);
      let url = 'http://localhost:5000/vehicle_plate/logs';
      if (filterDate) {
        url += `?date=${filterDate}`;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to fetch vehicle logs');
      }
      const data = await response.json();
      setVehicleLogs(data);
      setLogsLoading(false);
    } catch (err) {
      setError(err.message);
      setLogsLoading(false);
    }
  };

  const handleFilterDateChange = (e) => {
    setFilterDate(e.target.value);
  };

  const applyDateFilter = () => {
    fetchVehicleLogs();
  };

  const clearDateFilter = () => {
    setFilterDate('');
    // Fetch all logs without date filter
    fetchVehicleLogs();
  };

  const handleDeleteRecord = async (id) => {
    if (window.confirm('Are you sure you want to delete this vehicle record?')) {
      try {
        const response = await fetch(`http://localhost:5000/vehicle_plate/delete/${id}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          throw new Error('Failed to delete record');
        }

        // Refresh the records and stats
        fetchVehicleRecords();
        fetchVehicleStats();
        alert('Vehicle record deleted successfully');
      } catch (err) {
        alert(`Error deleting record: ${err.message}`);
      }
    }
  };

  const handleUpdateRecord = async (id, updatedData) => {
    try {
      const response = await fetch(`http://localhost:5000/vehicle_plate/update/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedData),
      });

      if (!response.ok) {
        throw new Error('Failed to update record');
      }

      // Refresh the records
      fetchVehicleRecords();
      alert('Vehicle record updated successfully');
    } catch (err) {
      alert(`Error updating record: ${err.message}`);
    }
  };

  // Filtering based on search term
  const filteredRecords = vehicleRecords.filter(record => 
    (record.plate_number && record.plate_number.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (record.owner && record.owner.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (record.model && record.model.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const filteredLogs = vehicleLogs.filter(log => 
    (log.plate_number && log.plate_number.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (log.owner && log.owner.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  // Pagination
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  
  const currentItems = activeTab === 'records' 
    ? filteredRecords.slice(indexOfFirstRecord, indexOfLastRecord)
    : filteredLogs.slice(indexOfFirstRecord, indexOfLastRecord);
    
  const totalPages = Math.ceil(
    (activeTab === 'records' ? filteredRecords.length : filteredLogs.length) / recordsPerPage
  );

  const [editingRecord, setEditingRecord] = useState(null);
  const [editForm, setEditForm] = useState({
    owner: '',
    model: ''
  });

  const startEditing = (record) => {
    setEditingRecord(record._id);
    setEditForm({
      owner: record.owner || '',
      model: record.model || ''
    });
  };

  const cancelEditing = () => {
    setEditingRecord(null);
  };

  const saveEdit = (id) => {
    // Map form field names to backend field names
    const updatedData = {
      owner_name: editForm.owner,
      vehicle_model: editForm.model
    };
    handleUpdateRecord(id, updatedData);
    setEditingRecord(null);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm({
      ...editForm,
      [name]: value
    });
  };

  // Helper function to format date safely
  const formatDate = (dateString) => {
    if (!dateString) {
      return 'Not Available';
    }
    try {
      return new Date(dateString).toLocaleString()
    } catch (e) {
      return 'Invalid Date';
    }
  };

  return (
    <div>
      <Navbar />
      <div className="dashboard-container">
        <h1>Vehicle Management Dashboard</h1>

        <div className="stats-container">
          <div className="stat-card">
            <h3>Total Vehicles</h3>
            <p>{stats.totalVehicles}</p>
          </div>
          <div className="stat-card">
            <h3>Registered Vehicles</h3>
            <p>{stats.registeredVehicles}</p>
          </div>
          <div className="stat-card">
            <h3>Detected Today</h3>
            <p>{stats.detectedToday}</p>
          </div>
          <div className="stat-card">
            <h3>Access Granted Today</h3>
            <p>{stats.accessGrantedToday}</p>
          </div>
        </div>

        <div className="tabs-container">
          <button 
            className={`tab-button ${activeTab === 'records' ? 'active' : ''}`}
            onClick={() => setActiveTab('records')}
          >
            Vehicle Records
          </button>
          <button 
            className={`tab-button ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Authentication Logs
          </button>
        </div>

        <div className="search-container">
          <input
            type="text"
            placeholder={`Search ${activeTab === 'records' ? 'records' : 'logs'}...`}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />

          {activeTab === 'logs' && (
            <div className="date-filter">
              <input
                type="date"
                value={filterDate}
                onChange={handleFilterDateChange}
                className="date-input"
              />
              <button onClick={applyDateFilter} className="filter-btn">Apply Filter</button>
              <button onClick={clearDateFilter} className="filter-btn clear-btn">Clear Filter</button>
            </div>
          )}
        </div>

        {(activeTab === 'records' && loading) || (activeTab === 'logs' && logsLoading) ? (
          <div className="loading">Loading...</div>
        ) : error ? (
          <div className="error">{error}</div>
        ) : (
          <>
            <div className="table-container">
              {activeTab === 'records' && (
                <table className="records-table">
                  <thead>
                    <tr>
                      <th>Plate Image</th>
                      <th>Plate Number</th>
                      <th>Owner</th>
                      <th>Vehicle Model</th>
                      <th>Registered At</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItems.length > 0 ? (
                      currentItems.map((record) => (
                        <tr key={record._id}>
                          <td>
                            {record.plate_image && (
                              <img 
                                src={`data:image/jpeg;base64,${record.plate_image}`} 
                                alt={record.plate_number} 
                                className="plate-thumbnail" 
                              />
                            )}
                          </td>
                          <td>{record.plate_number}</td>
                          <td>
                            {editingRecord === record._id ? (
                              <input
                                type="text"
                                name="owner"
                                value={editForm.owner}
                                onChange={handleInputChange}
                              />
                            ) : (
                              record.owner || 'Unknown'
                            )}
                          </td>
                          <td>
                            {editingRecord === record._id ? (
                              <input
                                type="text"
                                name="model"
                                value={editForm.model}
                                onChange={handleInputChange}
                              />
                            ) : (
                              record.model || 'Unknown'
                            )}
                          </td>
                          <td>{formatDate(record.registered_at)}</td>
                          <td>
                            {editingRecord === record._id ? (
                              <div className="action-buttons">
                                <button onClick={() => saveEdit(record._id)} className="save-btn">
                                  Save
                                </button>
                                <button onClick={cancelEditing} className="cancel-btn">
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <div className="action-buttons">
                                <button onClick={() => startEditing(record)} className="edit-btn">
                                  Edit
                                </button>
                                <button onClick={() => handleDeleteRecord(record._id)} className="delete-btn">
                                  Delete
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="6">No records found</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              )}

              {activeTab === 'logs' && (
                <table className="logs-table">
                  <thead>
                    <tr>
                      <th>Plate Number</th>
                      <th>Owner</th>
                      <th>Vehicle Model</th>
                      <th>Access Status</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentItems.length > 0 ? (
                      currentItems.map((log) => (
                        <tr key={log._id}>
                          <td>{log.plate_number}</td>
                          <td>{log.owner || 'Unknown'}</td>
                          <td>{log.model || 'Unknown'}</td>
                          <td>
                            <span className={`status-badge ${log.access_granted ? 'authorized' : 'unauthorized'}`}>
                              {log.access_granted ? 'Access Granted' : 'Access Denied'}
                            </span>
                          </td>
                          <td>{formatDate(log.timestamp)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="5">No logs found</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>

            {currentItems.length > 0 && totalPages > 1 && (
              <div className="pagination">
                <button
                  onClick={() => setCurrentPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="pagination-btn"
                >
                  Previous
                </button>
                <span>
                  Page {currentPage} of {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="pagination-btn"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default VehicleRecordsDashboard;