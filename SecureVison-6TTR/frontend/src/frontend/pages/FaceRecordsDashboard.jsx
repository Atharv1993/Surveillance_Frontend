import React, { useState, useEffect } from 'react';
import Navbar from '../../frontend/Navbar/Navbar';
import './Dashboard.css';

const FaceRecordsDashboard = () => {
  const [faceRecords, setFaceRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [recordsPerPage] = useState(10);
  const [stats, setStats] = useState({
    totalRecords: 0,
    attendanceToday: 0,
    knownFaces: 0,
    unknownFaces: 0
  });
  
  // New states for attendance logs
  const [attendanceLogs, setAttendanceLogs] = useState([]);
  const [attendanceLoading, setAttendanceLoading] = useState(false);
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [activeTab, setActiveTab] = useState('records'); // 'records' or 'attendance'

  useEffect(() => {
    fetchFaceRecords();
    fetchFaceStats();
  }, []);

  const fetchFaceRecords = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5000/face_recog/records');
      if (!response.ok) {
        throw new Error('Failed to fetch face records');
      }
      const data = await response.json();
      setFaceRecords(data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchFaceStats = async () => {
    try {
      const response = await fetch('http://localhost:5000/face_recog/stats');
      if (!response.ok) {
        throw new Error('Failed to fetch face stats');
      }
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching face stats:', err);
    }
  };

  // New function to fetch attendance logs
  const fetchAttendanceLogs = async () => {
    try {
      setAttendanceLoading(true);
      const response = await fetch(
        `http://localhost:5000/face_recog/attendance/logs?start_date=${startDate}&end_date=${endDate}`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch attendance logs');
      }
      const data = await response.json();
      setAttendanceLogs(data.logs || []);
      setAttendanceLoading(false);
    } catch (err) {
      console.error('Error fetching attendance logs:', err);
      setAttendanceLoading(false);
    }
  };

  const handleDeleteRecord = async (id) => {
    if (window.confirm('Are you sure you want to delete this record? This will remove the face from the recognition system.')) {
      try {
        const response = await fetch(`http://localhost:5000/face_recog/delete/${id}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          throw new Error('Failed to delete record');
        }

        // Refresh the records and stats
        fetchFaceRecords();
        fetchFaceStats();
        alert('Record deleted successfully');
      } catch (err) {
        alert(`Error deleting record: ${err.message}`);
      }
    }
  };

  const handleUpdateRecord = async (id, updatedData) => {
    try {
      const response = await fetch(`http://localhost:5000/face_recog/update/${id}`, {
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
      fetchFaceRecords();
      alert('Record updated successfully');
    } catch (err) {
      alert(`Error updating record: ${err.message}`);
    }
  };

  // Filtering based on search term - adjusted to handle missing fields safely
  const filteredRecords = faceRecords.filter(record => 
    (record.name && record.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (record.id_type && record.id_type.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (record.face_id && record.face_id.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  // Pagination
  const indexOfLastRecord = currentPage * recordsPerPage;
  const indexOfFirstRecord = indexOfLastRecord - recordsPerPage;
  const currentRecords = filteredRecords.slice(indexOfFirstRecord, indexOfLastRecord);
  const totalPages = Math.ceil(filteredRecords.length / recordsPerPage);

  const [editingRecord, setEditingRecord] = useState(null);
  const [editForm, setEditForm] = useState({
    name: '',
    id_type: ''
  });

  const startEditing = (record) => {
    setEditingRecord(record._id);
    setEditForm({
      name: record.name || '',
      id_type: record.id_type || ''
    });
  };

  const cancelEditing = () => {
    setEditingRecord(null);
  };

  const saveEdit = (id) => {
    handleUpdateRecord(id, editForm);
    setEditingRecord(null);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm({
      ...editForm,
      [name]: value
    });
  };

  // Helper function to format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch (e) {
      return 'Invalid Date';
    }
  };

  return (
    <div>
      <Navbar />
      <div className="dashboard-container">
        <h1>Face Recognition Dashboard</h1>

        <div className="stats-container">
          <div className="stat-card">
            <h3>Total Records</h3>
            <p>{stats.totalRecords}</p>
          </div>
          <div className="stat-card">
            <h3>Today's Attendance</h3>
            <p>{stats.attendanceToday}</p>
          </div>
          <div className="stat-card">
            <h3>Known Faces</h3>
            <p>{stats.knownFaces}</p>
          </div>
          {stats.unknownFaces > 0 && (
            <div className="stat-card">
              <h3>Unknown Faces</h3>
              <p>{stats.unknownFaces}</p>
            </div>
          )}
        </div>

        <div className="tabs-container">
          <button 
            className={`tab-button ${activeTab === 'records' ? 'active' : ''}`}
            onClick={() => setActiveTab('records')}
          >
            Face Records
          </button>
          <button 
            className={`tab-button ${activeTab === 'attendance' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('attendance');
              fetchAttendanceLogs();
            }}
          >
            Attendance Logs
          </button>
        </div>

        {activeTab === 'records' ? (
          // Face Records Tab
          <>
            <div className="search-container">
              <input
                type="text"
                placeholder="Search by name or ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
            </div>

            {loading ? (
              <div className="loading">Loading...</div>
            ) : error ? (
              <div className="error">{error}</div>
            ) : (
              <>
                <div className="table-container">
                  <table className="records-table">
                    <thead>
                      <tr>
                        <th>Image</th>
                        <th>Name</th>
                        <th>ID Type</th>
                        <th>Face ID</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentRecords.length > 0 ? (
                        currentRecords.map((record) => (
                          <tr key={record._id}>
                            <td>
                              {record.image ? (
                                <img 
                                  src={`data:image/jpeg;base64,${record.image}`} 
                                  alt={record.name} 
                                  className="face-thumbnail" 
                                />
                              ) : (
                                <div className="no-image">No Image</div>
                              )}
                            </td>
                            <td>
                              {editingRecord === record._id ? (
                                <input
                                  type="text"
                                  name="name"
                                  value={editForm.name}
                                  onChange={handleInputChange}
                                />
                              ) : (
                                record.name || 'N/A'
                              )}
                            </td>
                            <td>
                              {editingRecord === record._id ? (
                                <input
                                  type="text"
                                  name="id_type"
                                  value={editForm.id_type}
                                  onChange={handleInputChange}
                                />
                              ) : (
                                record.id_type || 'N/A'
                              )}
                            </td>
                            <td>{record.face_id || 'N/A'}</td>
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
                          <td colSpan="5">No records found</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {filteredRecords.length > recordsPerPage && (
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
          </>
        ) : (
          // Attendance Logs Tab
          <>
            <div className="date-filter-container">
              <div className="date-input-group">
                <label>Start Date:</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="date-input"
                />
              </div>
              <div className="date-input-group">
                <label>End Date:</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="date-input"
                />
              </div>
              <button 
                onClick={fetchAttendanceLogs} 
                className="filter-btn"
              >
                Filter
              </button>
            </div>

            {attendanceLoading ? (
              <div className="loading">Loading attendance logs...</div>
            ) : (
              <div className="table-container">
                <table className="records-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>ID/Roll</th>
                      <th>Date & Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attendanceLogs.length > 0 ? (
                      attendanceLogs.map((log) => (
                        <tr key={log._id}>
                          <td>{log.name || 'N/A'}</td>
                          <td>{log.roll || 'N/A'}</td>
                          <td>{log.timestamp || 'N/A'}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="3">No attendance logs found for selected date range</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default FaceRecordsDashboard;