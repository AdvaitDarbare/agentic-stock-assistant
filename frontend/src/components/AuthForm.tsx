'use client'

import { useState } from 'react'
import { Eye, EyeOff, User, Mail, Lock } from 'lucide-react'

interface AuthFormProps {
  onLogin: (email: string, password: string) => Promise<void>
  onRegister: (data: {
    email: string
    username: string
    password: string
    first_name: string
    last_name: string
  }) => Promise<void>
  isLoading: boolean
  error: string | null
}

export default function AuthForm({ onLogin, onRegister, isLoading, error }: AuthFormProps) {
  const [isLogin, setIsLogin] = useState(true)
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
    first_name: '',
    last_name: ''
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (isLogin) {
      await onLogin(formData.email, formData.password)
    } else {
      await onRegister(formData)
    }
  }

  const updateField = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  return (
    <div className="min-h-screen bg-[#2d3748] flex items-center justify-center px-4">
      <div className="max-w-md w-full">
        {/* Logo/Brand */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-[#c7700a] rounded-full mx-auto mb-4 flex items-center justify-center">
            <span className="text-white font-bold text-xl">F</span>
          </div>
          <h1 className="text-3xl font-bold text-[#f7fafc] mb-2">FinanceScope</h1>
          <p className="text-[#a0aec0]">AI-powered financial analysis and insights</p>
        </div>

        {/* Auth Form */}
        <div className="bg-[#374151] rounded-lg shadow-xl p-6 border border-[#4a5568]">
          <div className="mb-6">
            <div className="flex rounded-lg bg-[#2d3748] p-1">
              <button
                type="button"
                onClick={() => setIsLogin(true)}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  isLogin
                    ? 'bg-[#c7700a] text-white'
                    : 'text-[#a0aec0] hover:text-[#f7fafc]'
                }`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => setIsLogin(false)}
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  !isLogin
                    ? 'bg-[#c7700a] text-white'
                    : 'text-[#a0aec0] hover:text-[#f7fafc]'
                }`}
              >
                Sign Up
              </button>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-900/50 border border-red-500 rounded-md">
              <p className="text-red-200 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-[#f7fafc] mb-1">
                Email
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-4 w-4 text-[#a0aec0]" />
                </div>
                <input
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) => updateField('email', e.target.value)}
                  className="w-full pl-10 pr-3 py-2 border border-[#4a5568] rounded-md bg-[#2d3748] text-[#f7fafc] placeholder-[#a0aec0] focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent"
                  placeholder="Enter your email"
                />
              </div>
            </div>

            {/* Username (Register only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-[#f7fafc] mb-1">
                  Username
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User className="h-4 w-4 text-[#a0aec0]" />
                  </div>
                  <input
                    type="text"
                    required={!isLogin}
                    value={formData.username}
                    onChange={(e) => updateField('username', e.target.value)}
                    className="w-full pl-10 pr-3 py-2 border border-[#4a5568] rounded-md bg-[#2d3748] text-[#f7fafc] placeholder-[#a0aec0] focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent"
                    placeholder="Choose a username"
                  />
                </div>
              </div>
            )}

            {/* First Name & Last Name (Register only) */}
            {!isLogin && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-[#f7fafc] mb-1">
                    First Name
                  </label>
                  <input
                    type="text"
                    value={formData.first_name}
                    onChange={(e) => updateField('first_name', e.target.value)}
                    className="w-full px-3 py-2 border border-[#4a5568] rounded-md bg-[#2d3748] text-[#f7fafc] placeholder-[#a0aec0] focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent"
                    placeholder="First name"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-[#f7fafc] mb-1">
                    Last Name
                  </label>
                  <input
                    type="text"
                    value={formData.last_name}
                    onChange={(e) => updateField('last_name', e.target.value)}
                    className="w-full px-3 py-2 border border-[#4a5568] rounded-md bg-[#2d3748] text-[#f7fafc] placeholder-[#a0aec0] focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent"
                    placeholder="Last name"
                  />
                </div>
              </div>
            )}

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-[#f7fafc] mb-1">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-4 w-4 text-[#a0aec0]" />
                </div>
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={formData.password}
                  onChange={(e) => updateField('password', e.target.value)}
                  className="w-full pl-10 pr-10 py-2 border border-[#4a5568] rounded-md bg-[#2d3748] text-[#f7fafc] placeholder-[#a0aec0] focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent"
                  placeholder="Enter your password"
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4 text-[#a0aec0] hover:text-[#f7fafc]" />
                  ) : (
                    <Eye className="h-4 w-4 text-[#a0aec0] hover:text-[#f7fafc]" />
                  )}
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-[#c7700a] hover:bg-[#a0590a] disabled:bg-[#9ca3af] text-white font-medium py-2 px-4 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:ring-offset-2 focus:ring-offset-[#374151]"
            >
              {isLoading ? (
                <div className="flex items-center justify-center">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  {isLogin ? 'Signing In...' : 'Creating Account...'}
                </div>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          {/* Demo Credentials (for testing) */}
          {isLogin && (
            <div className="mt-4 p-3 bg-[#2d3748] rounded-md border border-[#4a5568]">
              <p className="text-xs text-[#a0aec0] mb-2">Demo Account:</p>
              <p className="text-xs text-[#a0aec0]">Email: user@example.com</p>
              <p className="text-xs text-[#a0aec0]">Password: securepassword123</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}